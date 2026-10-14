#include "fa_teleop/mujoco_engine.hpp"
#include "fa_teleop/planner.hpp"
#include <fstream>
#include <iomanip>
#include <iostream>
int main(int argc, char **argv) {
  try {
    if (argc < 2)
      throw std::runtime_error("bundle required");
    const std::string bundle = argv[1];
    fa::verify_bundle(bundle);
    fa::MujocoEngine engine(bundle);
    auto config = fa::Config::load(bundle + "/config/precision_200hz.yaml");
    fa::Kinematics kinematics(bundle +
                                  "/models/fa_description/urdf/fa_robot.urdf",
                              bundle + "/models/fa_robot.srdf", config);
    int checks = 0;
    auto require = [&](bool v, const char *message) {
      ++checks;
      if (!v)
        throw std::runtime_error(message);
    };
    require(engine.positions().norm() < 1e-12, "neutral startup");
    double max_fk_position = 0, max_fk_orientation = 0;
    for (int n = 0; n < 200; ++n) {
      auto q = fa::initial_pose();
      for (int j = 0; j < 14; ++j)
        q[j] += 0.15 * std::sin(n * 0.07 + j * 0.3);
      q[14] = 0.2 * std::sin(n * 0.04);
      q[15] = 0.1 * std::cos(n * 0.05);
      require(engine.apply(q), "apply valid upper command");
      engine.forward(0.001);
      require((engine.positions() - q).norm() < 1e-12,
              "all 16 joints execute directly, without interpolation");
      for (int side = 0; side < 2; ++side) {
        const auto p = engine.pose(side == 0),
                   fk = kinematics.fk(q.segment<7>(7 * side), side == 0);
        max_fk_position = std::max(max_fk_position, (p.p - fk.p).norm());
        max_fk_orientation =
            std::max(max_fk_orientation, fa::rotation_error(p.R, fk.R));
      }
    }
    require(max_fk_position < 1e-9 && max_fk_orientation < 1e-9,
            "MuJoCo and Pinocchio FK agree");
    const auto previous = engine.positions();
    auto invalid = previous;
    invalid[0] = 100;
    require(!engine.apply(invalid), "reject joint-limit violation");
    invalid = previous;
    invalid[1] = std::numeric_limits<double>::quiet_NaN();
    require(!engine.apply(invalid), "reject NaN command");
    require((engine.positions() - previous).norm() < 1e-12,
            "invalid input preserves last state");
    const int finger =
        mj_name2id(engine.model(), mjOBJ_JOINT, "left_index_mcp_pitch_joint");
    require(finger >= 0, "finger exists");
    engine.hand(true, 20);
    const double closed =
        engine.data()->qpos[engine.model()->jnt_qposadr[finger]];
    engine.hand(true, 21);
    require(closed > engine.data()->qpos[engine.model()->jnt_qposadr[finger]],
            "hand close/open display");
    fa::Upper seed = fa::initial_pose();
    engine.apply(seed);
    engine.forward(0.001);
    fa::Plan plan;
    std::array<double, 2> max_position{}, max_orientation{};
    int failures = 0;
    double max_work_ms = 0;
    for (int n = 0; n < 4000; ++n) {
      const double t = n * 0.005;
      auto goal = fa::initial_pose();
      for (int j = 0; j < 14; ++j)
        goal[j] += 0.02 * (std::sin(0.628318530718 * t + 0.15 * (j % 7)) -
                           std::sin(0.15 * (j % 7)));
      const std::array<fa::Pose, 2> desired{
          kinematics.fk(goal.head<7>(), true),
          kinematics.fk(goal.segment<7>(7), false)};
      const double started = fa::steady_seconds();
      const auto solved = kinematics.solve(desired, seed);
      fa::Target target;
      target.mode = 2;
      target.q = solved.q;
      target.duration = config.duration;
      target.velocity = config.velocity_limit;
      target.acceleration = config.acceleration_limit;
      const double activation = t + config.plan_lead;
      const auto from =
          plan.valid ? plan.sample(activation) : fa::stationary(seed);
      fa::Plan next;
      if (!solved.valid ||
          !fa::build_plan(from, target, config, activation, next)) {
        ++failures;
        continue;
      }
      seed = solved.q;
      bool switched = false;
      for (int k = 0; k < 5; ++k) {
        const double stamp = t + k * 0.001;
        if (stamp >= activation) {
          plan = next;
          switched = true;
        }
        const auto reference =
            plan.valid ? plan.sample(stamp) : fa::stationary(seed);
        require(engine.apply(reference.q), "valid sampled MuJoCo command");
        engine.forward(0.001);
      }
      if (!switched)
        plan = next;
      max_work_ms =
          std::max(max_work_ms, (fa::steady_seconds() - started) * 1000);
      for (int side = 0; side < 2; ++side) {
        const auto actual = engine.pose(side == 0);
        max_position[side] =
            std::max(max_position[side], (actual.p - desired[side].p).norm());
        max_orientation[side] =
            std::max(max_orientation[side],
                     fa::rotation_error(actual.R, desired[side].R));
      }
    }
    require(failures == 0, "MuJoCo tracking solver/planner failure");
    require(max_position[0] <= .001 && max_position[1] <= .001 &&
                max_orientation[0] <= .01 && max_orientation[1] <= .01,
            "MuJoCo model tracking precision");
    if (argc > 2) {
      std::ofstream out(argv[2]);
      out << std::setprecision(15) << "{\"release_id\":\"" << fa::release_id()
          << "\",\"engine\":\"" << mj_versionString()
          << "\",\"passed\":true,\"steps\":4000,\"fk_position_max_m\":"
          << max_fk_position
          << ",\"fk_orientation_max_rad\":" << max_fk_orientation
          << ",\"position_max_m\":[" << max_position[0] << ','
          << max_position[1] << "],\"orientation_max_rad\":["
          << max_orientation[0] << ',' << max_orientation[1]
          << "],\"max_work_ms\":" << max_work_ms << ",\"failures\":" << failures
          << ",\"physical_hardware\":false}\n";
    }
    std::cout << "PASS MuJoCo " << mj_versionString() << " checks=" << checks
              << " FK max=" << max_fk_position
              << " m, tracking position=" << max_position[0] << ','
              << max_position[1] << " m\n";
    return 0;
  } catch (const std::exception &e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
