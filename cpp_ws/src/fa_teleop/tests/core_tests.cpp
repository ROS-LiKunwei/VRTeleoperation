#include "fa_teleop/core.hpp"
#include "fa_teleop/planner.hpp"
#include <iostream>
#include <sstream>
#include <thread>

namespace {
int checks = 0;
void require(bool condition, const char *message) {
  ++checks;
  if (!condition)
    throw std::runtime_error(message);
}
std::array<Eigen::Vector3d, 26> points() {
  std::array<Eigen::Vector3d, 26> p;
  for (auto &x : p)
    x = Eigen::Vector3d(0, 0, 0);
  p[7] = {0.03, 0.08, 0};
  p[12] = {0, 0.09, 0};
  p[22] = {-0.03, 0.07, 0};
  return p;
}
std::string payload() {
  std::ostringstream s;
  s << "relative:";
  auto p = points();
  for (size_t i = 0; i < 26; ++i) {
    if (i)
      s << '|';
    s << p[i].x() << ',' << p[i].y() << ',' << p[i].z();
  }
  s << ":hand_command=2";
  return s.str();
}
} // namespace
int main(int argc, char **argv) {
  try {
    if (argc < 2)
      throw std::runtime_error("bundle directory required");
    fa::verify_bundle(argv[1]);
    if (argc > 2 && std::string(argv[2]) == "--verify-only")
      return 0;
    fa::Config c =
        fa::Config::load(std::string(argv[1]) + "/config/precision_200hz.yaml");
    require(c.control_hz == 200 && c.sample_hz == 1000, "configured timing");
    fa::HandSample h;
    require(fa::parse_hand(payload(), false, 100, 1000, 1, h),
            "legacy hand payload");
    require(h.valid && h.relative && h.command == 2, "parsed hand flags");
    require(std::abs(h.pose.R.determinant() - 1) < 1e-12,
            "right-handed hand frame");
    fa::HandSample left;
    require(fa::parse_hand(payload(), true, 100, 1000, 1, left),
            "left hand payload");
    require((left.pose.R.col(1) + h.pose.R.col(1)).norm() < 1e-12,
            "left palm normal convention");
    require(!fa::parse_hand("relative:0,0,0", false, 100, 1000, 1, h),
            "reject truncated frame");
    require(!fa::parse_hand(payload() + "|1,2,3", false, 100, 1000, 1, h),
            "reject malformed metadata");
    auto nan = payload();
    nan.replace(nan.find("0,0,0"), 1, "nan");
    require(!fa::parse_hand(nan, false, 100, 1000, 1, h), "reject NaN");
    auto bad_time = "99:00:00.000000:" + payload();
    require(!fa::parse_hand(bad_time, false, 100, 1000, 1, h),
            "reject invalid timestamp");
    require(!fa::parse_hand(std::string(5000, 'x'), false, 100, 1000, 1, h),
            "bound frame size");
    fa::Pose baseline, current, robot;
    robot.p = {0.3, 0.2, 0.5};
    current.R =
        Eigen::AngleAxisd(0.3, Eigen::Vector3d::UnitY()).toRotationMatrix();
    const auto mapped =
        fa::retarget(current, baseline, robot, fa::nominal_mapping(), 1);
    require((mapped.p - robot.p).norm() < 1e-12,
            "pure wrist rotation must not move TCP");
    current.p = {0.01, 0, 0};
    require(
        (fa::retarget(current, baseline, robot, fa::nominal_mapping(), 1).p -
         robot.p - Eigen::Vector3d(0, -0.01, 0))
                .norm() < 1e-12,
        "FA axis mapping");
    require(fa::fresh(100, 1000, 100.01, 1000.01, 0.05, c), "fresh sample");
    require(!fa::fresh(100, 999, 100.01, 1000.01, 0.05, c),
            "reject old source stamp");
    require(!fa::fresh(100, 1001, 100.01, 1000.01, 0.05, c),
            "reject future source stamp");
    require(!fa::fresh(100, 0, 100.01, 1000.01, 0.05, c),
            "require source stamp");
    std::vector<std::string> names(fa::joint_names().begin(),
                                   fa::joint_names().end());
    auto q = fa::initial_pose();
    std::vector<double> positions(q.data(), q.data() + 16);
    fa::Feedback fb;
    require(fa::decode_feedback(names, positions, {}, 1000, 100, fb),
            "complete feedback");
    require(!fa::decode_feedback(names, positions, {}, 1000, 101, fb),
            "reject repeated source feedback");
    names.pop_back();
    positions.pop_back();
    require(!fa::decode_feedback(names, positions, {}, 1001, 101, fb),
            "reject partial feedback");
    auto candidate = q;
    candidate[0] += 0.01;
    require(fa::safe_target(candidate, q, fb, c), "safe target");
    candidate[0] += 1;
    require(!fa::safe_target(candidate, q, fb, c), "reject joint jump");
    candidate = q;
    candidate[0] = 10;
    require(!fa::safe_target(candidate, q, fb, c),
            "reject joint limit violation");
    fa::Config bad = c;
    bad.ik_position = 0.1;
    bool rejected = false;
    try {
      bad.validate();
    } catch (...) {
      rejected = true;
    }
    require(rejected, "reject relaxed precision configuration");
    fa::SpscQueue<int, 4> queue;
    require(queue.push(1) && queue.push(2) && queue.push(3) && !queue.push(4),
            "bounded event queue");
    int event = 0;
    require(queue.pop(event) && event == 1, "ordered event queue");
    fa::Snapshot<fa::Feedback> snapshot;
    std::atomic<bool> done{false};
    std::thread producer([&] {
      for (int n = 0; n < 20000; ++n) {
        fa::Feedback f;
        f.sequence = n;
        f.q.setConstant(n);
        snapshot.write(f);
      }
      done = true;
    });
    int race_errors = 0;
    while (!done) {
      fa::Feedback f;
      if (snapshot.read(f) && f.q.minCoeff() != f.q.maxCoeff())
        ++race_errors;
    }
    producer.join();
    require(race_errors == 0, "race-free snapshots");
    fa::AxisCalibration calibration(c);
    fa::HandSample l, r;
    l.valid = r.valid = true;
    double t = 1;
    uint64_t seq = 0;
    auto stable = [&](Eigen::Vector3d delta) {
      for (int n = 0; n < 100; ++n) {
        t += 0.01;
        l.received = r.received = t;
        l.sequence = r.sequence = ++seq;
        l.pose.p = Eigen::Vector3d(-0.2, 0, 0) + delta;
        r.pose.p = Eigen::Vector3d(0.2, 0, 0) + delta;
        calibration.update(l, r, t);
      }
    };
    stable(Eigen::Vector3d::Zero());
    require(calibration.state() == fa::CalibrationState::Forward,
            "calibration origin");
    stable({0.1, 0, 0});
    require(calibration.state() == fa::CalibrationState::ReturnForward,
            "calibration forward");
    stable({0, 0, 0});
    require(calibration.state() == fa::CalibrationState::Left,
            "calibration return");
    stable({0, 0.1, 0});
    require(calibration.state() == fa::CalibrationState::ReturnReady,
            "calibration lateral");
    stable({0, 0, 0});
    require(calibration.state() == fa::CalibrationState::Ready,
            "calibration ready");
    require((calibration.mapping() - Eigen::Matrix3d::Identity()).norm() <
                1e-10,
            "calibrated basis");
    fa::Target target;
    target.mode = 2;
    target.sequence = 1;
    target.q = q;
    target.q[0] += 0.001;
    target.duration = 0.04;
    target.velocity = c.velocity_limit;
    target.acceleration = c.acceleration_limit;
    fa::Plan plan;
    require(fa::build_plan(fa::stationary(q), target, c, 1, plan),
            "bounded minimum-snap plan");
    const auto start = plan.sample(1), end = plan.sample(1 + plan.duration + 1);
    require((start.q - q).norm() < 1e-10 && start.velocity.norm() < 1e-9,
            "trajectory start continuity");
    require((end.q - target.q).norm() < 1e-9 && end.velocity.norm() < 1e-8,
            "trajectory target endpoint");
    for (int n = 0; n < 100; ++n)
      require(fa::valid_reference(plan.sample(1 + plan.duration * n / 99), c),
              "trajectory bounds");
    const double at = 1 + plan.duration / 2;
    const auto moving = plan.sample(at);
    target.mode = 0;
    target.sequence = 2;
    fa::Plan braking;
    require(fa::build_plan(moving, target, c, at, braking),
            "controlled brake plan");
    require((braking.sample(at).velocity - moving.velocity).norm() < 1e-7,
            "braking velocity continuity");
    require(braking.sample(at + braking.duration + 1).velocity.norm() < 1e-8,
            "braking stops");
    fa::Kinematics kin(std::string(argv[1]) +
                           "/models/fa_description/urdf/fa_robot.urdf",
                       std::string(argv[1]) + "/models/fa_robot.srdf", c);
    std::array<fa::Pose, 2> targets{kin.fk(q.head<7>(), true),
                                    kin.fk(q.segment<7>(7), false)};
    auto solve = kin.solve(targets, q);
    require(solve.valid, "FK/IK round trip");
    targets[0].p.x() += 10;
    solve = kin.solve(targets, q);
    require(!solve.valid, "reject unreachable target");
    std::cout << "PASS " << checks << " contract checks\n";
    return 0;
  } catch (const std::exception &e) {
    std::cerr << "FAIL: " << e.what() << '\n';
    return 1;
  }
}
