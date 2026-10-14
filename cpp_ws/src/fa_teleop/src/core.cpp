#include "fa_teleop/core.hpp"
#include <algorithm>
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <sstream>
#include <stdexcept>
#include <type_traits>
#include <yaml-cpp/yaml.h>

namespace fa {
double steady_seconds() {
  return std::chrono::duration<double>(Clock::now().time_since_epoch()).count();
}
double wall_seconds() {
  return std::chrono::duration<double>(
             std::chrono::system_clock::now().time_since_epoch())
      .count();
}
Config Config::load(const std::string &path) {
  Config c;
  c.profile_sha = file_sha256(path);
  const auto n = YAML::LoadFile(path);
  auto read = [&](const char *key, auto &value) {
    if (n[key])
      value = n[key].as<std::decay_t<decltype(value)>>();
  };
  read("control_hz", c.control_hz);
  read("sample_hz", c.sample_hz);
  read("input_timeout_s", c.input_timeout);
  read("feedback_timeout_s", c.feedback_timeout);
  read("source_max_age_s", c.source_max_age);
  read("timestamp_skew_s", c.timestamp_skew);
  read("require_source_stamp", c.require_source_stamp);
  read("calibration_enabled", c.calibration_enabled);
  read("live_enabled", c.live_enabled);
  read("ik_iterations", c.ik_iterations);
  read("ik_budget_ms", c.ik_budget_ms);
  read("ik_eps", c.ik_eps);
  read("ik_position_m", c.ik_position);
  read("ik_orientation_rad", c.ik_orientation);
  read("max_joint_jump_rad", c.max_joint_jump);
  read("max_tracking_error_rad", c.max_tracking_error);
  read("velocity_limit_rad_s", c.velocity_limit);
  read("acceleration_limit_rad_s2", c.acceleration_limit);
  read("trajectory_duration_s", c.duration);
  read("plan_lead_s", c.plan_lead);
  read("target_timeout_s", c.target_timeout);
  read("translation_scale", c.translation_scale);
  read("initialization_duration_s", c.initialization_duration);
  read("initialization_velocity_rad_s", c.initialization_velocity);
  read("initialization_plan_lead_s", c.initialization_plan_lead);
  read("calibration_dwell_s", c.calibration_dwell);
  read("calibration_stability_m", c.calibration_stability);
  read("calibration_min_motion_m", c.calibration_min_motion);
  read("calibration_max_motion_m", c.calibration_max_motion);
  read("calibration_return_m", c.calibration_return);
  read("origin_jump_m", c.origin_jump_m);
  read("origin_jump_rad", c.origin_jump_rad);
  read("constraint_samples", c.constraint_samples);
  read("duration_search_iterations", c.duration_search_iterations);
  read("snap_weight", c.snap_weight);
  read("max_plan_duration_s", c.max_plan_duration);
  read("bind_address", c.bind_address);
  read("right_port", c.right_port);
  read("left_port", c.left_port);
  read("event_port", c.event_port);
  read("button_port", c.button_port);
  read("prompt_port", c.prompt_port);
  read("hand_open_command", c.hand_open);
  read("hand_close_command", c.hand_close);
  c.validate();
  return c;
}
void Config::validate() const {
  for (double v : {control_hz,
                   sample_hz,
                   input_timeout,
                   feedback_timeout,
                   source_max_age,
                   timestamp_skew,
                   ik_budget_ms,
                   ik_eps,
                   ik_position,
                   ik_orientation,
                   max_joint_jump,
                   max_tracking_error,
                   velocity_limit,
                   acceleration_limit,
                   duration,
                   plan_lead,
                   target_timeout,
                   translation_scale,
                   initialization_duration,
                   initialization_velocity,
                   initialization_plan_lead,
                   calibration_dwell,
                   calibration_stability,
                   calibration_min_motion,
                   calibration_max_motion,
                   calibration_return,
                   origin_jump_m,
                   origin_jump_rad,
                   max_plan_duration})
    if (!std::isfinite(v) || v <= 0)
      throw std::invalid_argument("Non-positive/non-finite configuration");
  if (control_hz != 200 || sample_hz != 1000 ||
      ik_budget_ms >= 1000 / control_hz || ik_position > 0.001 ||
      ik_orientation > 0.01 || ik_iterations < 1 || ik_iterations > 100 ||
      constraint_samples < 20 || constraint_samples > 1000 ||
      duration_search_iterations < 1 || duration_search_iterations > 20 ||
      !std::isfinite(snap_weight) || snap_weight < 0 || snap_weight > 1 ||
      calibration_min_motion >= calibration_max_motion ||
      plan_lead >= target_timeout || initialization_plan_lead < plan_lead ||
      initialization_plan_lead > 0.25)
    throw std::invalid_argument(
        "Configuration violates FA timing/precision contract");
  std::array<int, 5> ports{right_port, left_port, event_port, button_port,
                           prompt_port};
  std::sort(ports.begin(), ports.end());
  if (ports.front() < 1024 || ports.back() > 65535 ||
      std::adjacent_find(ports.begin(), ports.end()) != ports.end())
    throw std::invalid_argument("Invalid/duplicate ZMQ port");
}
const std::array<std::string, 16> &joint_names() {
  static const std::array<std::string, 16> names{"left_shoulder_pitch_joint",
                                                 "left_shoulder_roll_joint",
                                                 "left_shoulder_yaw_joint",
                                                 "left_elbow_joint",
                                                 "left_wrist_yaw_joint",
                                                 "left_wrist_pitch_joint",
                                                 "left_wrist_roll_joint",
                                                 "right_shoulder_pitch_joint",
                                                 "right_shoulder_roll_joint",
                                                 "right_shoulder_yaw_joint",
                                                 "right_elbow_joint",
                                                 "right_wrist_yaw_joint",
                                                 "right_wrist_pitch_joint",
                                                 "right_wrist_roll_joint",
                                                 "neck_yaw_joint",
                                                 "neck_pitch_joint"};
  return names;
}
Upper lower_limits() {
  Upper q;
  q << -2.750, -0.288, -2.650, -2.050, -2.735, -0.469, -1.550, -2.750, -3.330,
      -2.570, -2.050, -2.730, -0.480, -1.536, -1.570, -0.523;
  return q;
}
Upper upper_limits() {
  Upper q;
  q << 2.748, 3.342, 2.583, 1.440, 2.729, 0.482, 1.520, 2.750, 0.302, 2.660,
      1.440, 2.730, 0.470, 1.532, 1.570, 0.785;
  return q;
}
Upper initial_pose() {
  Upper q;
  q << -1.05, 0.76, -0.62, -1.03, -0.68, 0, -0.35, -1.05, -0.76, 0.62, -1.03,
      0.68, 0, 0.35, 0, 0;
  return q;
}
Eigen::Matrix3d nominal_mapping() {
  Eigen::Matrix3d r;
  r << 0, 0, -1, -1, 0, 0, 0, 1, 0;
  return r;
}
double rotation_error(const Eigen::Matrix3d &a, const Eigen::Matrix3d &b) {
  return std::abs(Eigen::AngleAxisd(a.transpose() * b).angle());
}
bool valid_hand_pose(const std::array<Eigen::Vector3d, 26> &p, bool left,
                     Pose &pose) {
  for (const auto &v : p)
    if (!v.allFinite())
      return false;
  const auto v1 = (p[7] - p[0]).eval(), v2 = (p[22] - p[0]).eval(),
             v3 = (p[12] - p[0]).eval();
  Eigen::Vector3d normal = v1.cross(v3), direction = (v1 + v2 + v3) / 3;
  if (left)
    normal = -normal;
  if (normal.norm() < 1e-8 || direction.norm() < 0.005 ||
      direction.norm() > 0.5)
    return false;
  normal.normalize();
  direction.normalize();
  Eigen::Vector3d x = direction.cross(normal);
  if (x.norm() < 1e-6)
    return false;
  x.normalize();
  Eigen::Vector3d y = normal - x.dot(normal) * x;
  if (y.norm() < 1e-6)
    return false;
  y.normalize();
  pose.p = p[0];
  pose.R.col(0) = x;
  pose.R.col(1) = y;
  pose.R.col(2) = x.cross(y);
  return pose.R.allFinite() && pose.R.determinant() > 0.999;
}
bool parse_hand(const std::string &raw, bool left, double received, double wall,
                uint64_t seq, HandSample &out) {
  if (raw.size() > 4096)
    return false;
  size_t mode = raw.find("relative:");
  bool relative = true;
  if (mode == std::string::npos) {
    mode = raw.find("absolute:");
    relative = false;
  }
  if (mode == std::string::npos)
    return false;
  HandSample h;
  h.received = received;
  h.sequence = seq;
  h.relative = relative;
  if (mode > 0) {
    int hour = 0, minute = 0;
    double sec = 0;
    char tail = 0;
    const auto stamp = raw.substr(0, mode - 1);
    if (std::sscanf(stamp.c_str(), "%d:%d:%lf%c", &hour, &minute, &sec,
                    &tail) != 3 ||
        hour < 0 || hour > 23 || minute < 0 || minute > 59 ||
        !std::isfinite(sec) || sec < 0 || sec >= 60)
      return false;
    // Unity sends local time of day. Use the closest day and require clock
    // synchronization at deployment.
    const time_t t = static_cast<time_t>(wall);
    std::tm local{};
    localtime_r(&t, &local);
    double delta = hour * 3600 + minute * 60 + sec -
                   (local.tm_hour * 3600 + local.tm_min * 60 + local.tm_sec +
                    (wall - std::floor(wall)));
    if (delta > 43200)
      delta -= 86400;
    if (delta < -43200)
      delta += 86400;
    h.source_wall = wall + delta;
  }
  std::string coords = raw.substr(mode + 9);
  auto meta = coords.find(":hand_command=");
  if (meta != std::string::npos) {
    auto command = coords.substr(meta + 14);
    while (!command.empty() &&
           (command.back() == ':' || std::isspace(command.back())))
      command.pop_back();
    if (command != "1" && command != "2")
      return false;
    h.command = command[0] - '0';
    coords.resize(meta);
  }
  while (!coords.empty() &&
         (coords.back() == ':' || std::isspace(coords.back())))
    coords.pop_back();
  std::replace(coords.begin(), coords.end(), ',', ' ');
  std::replace(coords.begin(), coords.end(), '|', ' ');
  std::istringstream stream(coords);
  for (auto &p : h.points) {
    if (!(stream >> p.x() >> p.y() >> p.z()) || !p.allFinite())
      return false;
    p.z() = -p.z();
  }
  std::string extra;
  if (stream >> extra)
    return false;
  if (!valid_hand_pose(h.points, left, h.pose))
    return false;
  h.valid = true;
  out = h;
  return true;
}
Pose retarget(const Pose &current, const Pose &baseline, const Pose &robot,
              const Eigen::Matrix3d &m, double scale) {
  Pose out;
  out.p = robot.p + m * (current.p - baseline.p) * scale;
  out.R = m * (current.R * baseline.R.transpose()) * m.transpose() * robot.R;
  return out;
}
bool fresh(double received, double source, double now, double wall,
           double timeout, const Config &c) {
  if (!std::isfinite(received) || received <= 0 || now < received ||
      now - received > timeout)
    return false;
  if (source <= 0)
    return !c.require_source_stamp;
  return std::isfinite(source) && wall - source >= -c.timestamp_skew &&
         wall - source <= c.source_max_age;
}
bool decode_feedback(const std::vector<std::string> &names,
                     const std::vector<double> &p, const std::vector<double> &v,
                     double stamp, double now, Feedback &output) {
  if (names.size() != p.size() || !std::isfinite(stamp) || stamp <= 0 ||
      stamp <= output.source_wall)
    return false;
  Feedback next;
  next.source_wall = stamp;
  next.received = now;
  next.sequence = output.sequence + 1;
  for (size_t i = 0; i < 16; ++i) {
    const auto it = std::find(names.begin(), names.end(), joint_names()[i]);
    // Neck must be supplied, rather than silently replacing its physical state
    // with zero.
    if (it == names.end() || std::count(names.begin(), names.end(), *it) != 1)
      return false;
    const size_t k = std::distance(names.begin(), it);
    if (!std::isfinite(p[k]))
      return false;
    next.q[i] = p[k];
    if (k < v.size() && std::isfinite(v[k]) && std::abs(v[k]) <= 2)
      next.velocity[i] = v[k];
  }
  if ((next.q.array() < lower_limits().array()).any() ||
      (next.q.array() > upper_limits().array()).any())
    return false;
  next.valid = true;
  output = next;
  return true;
}
bool safe_target(const Upper &candidate, const Upper &previous,
                 const Feedback &fb, const Config &c) {
  return fb.valid && candidate.allFinite() && previous.allFinite() &&
         (candidate.array() >= lower_limits().array()).all() &&
         (candidate.array() <= upper_limits().array()).all() &&
         (candidate - previous).cwiseAbs().maxCoeff() <= c.max_joint_jump &&
         (previous - fb.q).head<14>().cwiseAbs().maxCoeff() <=
             c.max_tracking_error;
}
void AxisCalibration::reset() {
  state_ = CalibrationState::Origin;
  mapping_ = nominal_mapping();
  stable_since_ = -1;
  count_ = 0;
  last_l_ = last_r_ = 0;
}
bool AxisCalibration::stable(const Eigen::Vector3d &l, const Eigen::Vector3d &r,
                             double now) {
  if (stable_since_ < 0 ||
      (l - stable_l_).norm() > config_.calibration_stability ||
      (r - stable_r_).norm() > config_.calibration_stability) {
    stable_since_ = now;
    stable_l_ = l;
    stable_r_ = r;
    sum_l_ = l;
    sum_r_ = r;
    count_ = 1;
    return false;
  }
  sum_l_ += l;
  sum_r_ += r;
  ++count_;
  return count_ >= 3 && now - stable_since_ >= config_.calibration_dwell;
}
bool AxisCalibration::motion_valid(const Eigen::Vector3d &l,
                                   const Eigen::Vector3d &r) const {
  const Eigen::Vector3d dl = l - origin_l_, dr = r - origin_r_,
                        d = (dl + dr) / 2;
  if (dl.norm() < 1e-6 || dr.norm() < 1e-6 ||
      d.norm() < config_.calibration_min_motion ||
      d.norm() > config_.calibration_max_motion)
    return false;
  const double cosine = std::cos(20.0 * 3.141592653589793 / 180);
  return dl.normalized().dot(d.normalized()) >= cosine &&
         dr.normalized().dot(d.normalized()) >= cosine &&
         std::abs(dl.norm() - dr.norm()) / std::max(dl.norm(), dr.norm()) <=
             0.35;
}
bool AxisCalibration::update(const HandSample &l, const HandSample &r,
                             double now) {
  if (state_ == CalibrationState::Ready)
    return true;
  if (state_ == CalibrationState::Failed || !l.valid || !r.valid ||
      l.sequence == last_l_ || r.sequence == last_r_ ||
      std::abs(l.received - r.received) > config_.timestamp_skew)
    return false;
  last_l_ = l.sequence;
  last_r_ = r.sequence;
  if (!stable(l.pose.p, r.pose.p, now))
    return false;
  const Eigen::Vector3d lp = sum_l_ / count_, rp = sum_r_ / count_,
                        d = (lp + rp - origin_l_ - origin_r_) / 2;
  auto advance = [&](CalibrationState s) {
    state_ = s;
    stable_since_ = -1;
    count_ = 0;
  };
  switch (state_) {
  case CalibrationState::Origin:
    origin_l_ = lp;
    origin_r_ = rp;
    advance(CalibrationState::Forward);
    break;
  case CalibrationState::Forward:
    if (d.norm() > config_.calibration_max_motion) {
      advance(CalibrationState::Failed);
      break;
    }
    if (motion_valid(lp, rp)) {
      forward_ = d.normalized();
      advance(CalibrationState::ReturnForward);
    }
    break;
  case CalibrationState::ReturnForward:
    if ((lp - origin_l_).norm() < config_.calibration_return &&
        (rp - origin_r_).norm() < config_.calibration_return)
      advance(CalibrationState::Left);
    break;
  case CalibrationState::Left:
    if (d.norm() > config_.calibration_max_motion) {
      advance(CalibrationState::Failed);
      break;
    }
    if (motion_valid(lp, rp)) {
      if (std::abs(forward_.dot(d.normalized())) >
          config_.calibration_max_dot) {
        advance(CalibrationState::Failed);
        break;
      }
      Eigen::Vector3d lateral = (d - forward_ * forward_.dot(d)).normalized();
      Eigen::Matrix3d basis;
      basis.col(0) = forward_;
      basis.col(1) = lateral;
      basis.col(2) = forward_.cross(lateral);
      mapping_ = basis.transpose();
      advance(CalibrationState::ReturnReady);
    }
    break;
  case CalibrationState::ReturnReady:
    if ((lp - origin_l_).norm() < config_.calibration_return &&
        (rp - origin_r_).norm() < config_.calibration_return)
      advance(CalibrationState::Ready);
    break;
  default:
    break;
  }
  return state_ == CalibrationState::Ready;
}
const char *AxisCalibration::prompt() const {
  switch (state_) {
  case CalibrationState::Origin:
    return "fa_calib_origin";
  case CalibrationState::Forward:
    return "fa_calib_forward";
  case CalibrationState::ReturnForward:
    return "fa_calib_return";
  case CalibrationState::Left:
    return "fa_calib_left";
  case CalibrationState::ReturnReady:
    return "fa_calib_return";
  case CalibrationState::Ready:
    return "fa_calib_success";
  default:
    return "fa_calib_failed";
  }
}
Kinematics::Kinematics(const std::string &model, const std::string &srdf,
                       const Config &c)
    : solver_(model, srdf), config_(c) {
  // Redundant posture forces must not trade task-space precision for comfort.
  solver_.setComfortNullspace(0, Arm::Zero(), Arm::Zero());
}
Pose Kinematics::fk(const Arm &q, bool left) {
  fa_arm_kinematic::ArmKinematicsOptions o;
  const auto p = solver_.computeArmFK_SE3(
      q,
      left ? fa_arm_kinematic::ArmSide::LEFT : fa_arm_kinematic::ArmSide::RIGHT,
      o);
  return {p.p, p.R};
}
SolveResult Kinematics::solve(const std::array<Pose, 2> &targets,
                              const Upper &seed) {
  SolveResult out;
  out.q = seed;
  const double started = steady_seconds();
  fa_arm_kinematic::ArmKinematicsOptions o;
  o.skip_svd_fallback = true;
  o.acceptable_position_error = config_.ik_position;
  o.acceptable_orientation_error = config_.ik_orientation;
  o.continuity_nullspace_weight = 0;
  o.deadline =
      Clock::now() +
      std::chrono::duration_cast<Clock::duration>(
          std::chrono::duration<double, std::milli>(config_.ik_budget_ms));
  out.valid = true;
  for (int i = 0; i < 2; ++i) {
    if (!targets[i].p.allFinite() || !targets[i].R.allFinite()) {
      out.valid = false;
      break;
    }
    const auto side = i == 0 ? fa_arm_kinematic::ArmSide::LEFT
                             : fa_arm_kinematic::ArmSide::RIGHT;
    const auto result = solver_.solveArmIK(
        pinocchio::SE3(targets[i].R, targets[i].p), side,
        seed.segment<7>(7 * i), o, config_.ik_iterations, config_.ik_eps);
    if (!result.has_solution || result.q_solution.size() != 7 ||
        !result.q_solution.allFinite()) {
      out.valid = false;
      break;
    }
    const auto actual = fk(result.q_solution, i == 0);
    out.position_error[i] = (actual.p - targets[i].p).norm();
    out.orientation_error[i] = rotation_error(actual.R, targets[i].R);
    if (out.position_error[i] > config_.ik_position ||
        out.orientation_error[i] > config_.ik_orientation)
      out.valid = false;
    out.q.segment<7>(7 * i) = result.q_solution;
    if (Clock::now() >= o.deadline) {
      out.valid = false;
      out.deadline_missed = true;
      break;
    }
  }
  out.elapsed_ms = (steady_seconds() - started) * 1000;
  return out;
}
} // namespace fa
