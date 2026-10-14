#pragma once
#include "ik_7dof/fa_ik_solver.hpp"
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <deque>
#include <mutex>
#include <optional>
#include <string>
#include <vector>

namespace fa {
using Arm = Eigen::Matrix<double, 7, 1>;
using Upper = Eigen::Matrix<double, 16, 1>;
using Clock = std::chrono::steady_clock;
double steady_seconds();
double wall_seconds();
struct Pose {
  Eigen::Vector3d p{Eigen::Vector3d::Zero()};
  Eigen::Matrix3d R{Eigen::Matrix3d::Identity()};
};
struct HandSample {
  std::array<Eigen::Vector3d, 26> points{};
  Pose pose;
  double received{0}, source_wall{0};
  uint64_t sequence{0};
  int command{0};
  bool relative{false}, valid{false};
};
struct Feedback {
  Upper q{Upper::Zero()}, velocity{Upper::Zero()};
  double received{0}, source_wall{0};
  uint64_t sequence{0};
  bool valid{false};
};
template <class T> class Snapshot {
public:
  void write(const T &value) {
    std::lock_guard<std::mutex> lock(mutex_);
    value_ = value;
  }
  bool read(T &value) const {
    std::unique_lock<std::mutex> lock(mutex_, std::try_to_lock);
    if (!lock)
      return false;
    value = value_;
    return true;
  }

private:
  mutable std::mutex mutex_;
  T value_{};
};
template <class T, size_t N> class SpscQueue {
public:
  bool push(const T &value) {
    const size_t w = write_.load(std::memory_order_relaxed), next = (w + 1) % N;
    if (next == read_.load(std::memory_order_acquire))
      return false;
    values_[w] = value;
    write_.store(next, std::memory_order_release);
    return true;
  }
  bool pop(T &value) {
    const size_t r = read_.load(std::memory_order_relaxed);
    if (r == write_.load(std::memory_order_acquire))
      return false;
    value = values_[r];
    read_.store((r + 1) % N, std::memory_order_release);
    return true;
  }

private:
  std::array<T, N> values_{};
  std::atomic<size_t> read_{0}, write_{0};
};
struct Config {
  std::string profile_sha;
  std::string control_backend{"real"};
  double control_hz{200}, sample_hz{1000}, input_timeout{0.1},
      feedback_timeout{0.05};
  double source_max_age{0.15}, timestamp_skew{0.05};
  bool require_source_stamp{true}, calibration_enabled{true},
      live_enabled{false};
  int ik_iterations{35};
  double ik_budget_ms{3}, ik_eps{0.0001}, ik_position{0.0002},
      ik_orientation{0.002};
  double max_joint_jump{0.15}, max_tracking_error{0.1}, velocity_limit{0.5},
      acceleration_limit{3};
  double duration{0.04}, plan_lead{0.004}, target_timeout{0.075},
      translation_scale{1};
  double initialization_duration{5}, initialization_velocity{0.3};
  double initialization_plan_lead{0.05};
  double calibration_dwell{0.4}, calibration_stability{0.02};
  double calibration_min_motion{0.06}, calibration_max_motion{0.6};
  double calibration_return{0.04}, calibration_max_dot{0.35};
  double origin_jump_m{0.15}, origin_jump_rad{0.2617993878};
  int constraint_samples{200}, duration_search_iterations{10};
  double snap_weight{0.85}, max_plan_duration{2};
  std::string bind_address{"*"};
  int right_port{8087}, left_port{8110}, event_port{8100}, button_port{8095},
      prompt_port{8112};
  int hand_open{21}, hand_close{20};
  static Config load(const std::string &path);
  void validate() const;
};
const std::array<std::string, 16> &joint_names();
Upper lower_limits();
Upper upper_limits();
Upper initial_pose();
Eigen::Matrix3d nominal_mapping();
double rotation_error(const Eigen::Matrix3d &a, const Eigen::Matrix3d &b);
bool parse_hand(const std::string &raw, bool left, double received, double wall,
                uint64_t seq, HandSample &output);
bool valid_hand_pose(const std::array<Eigen::Vector3d, 26> &internal, bool left,
                     Pose &pose);
Pose retarget(const Pose &current, const Pose &baseline,
              const Pose &robot_baseline, const Eigen::Matrix3d &mapping,
              double scale);
bool fresh(double received, double source, double now, double wall,
           double rx_timeout, const Config &c);
bool decode_feedback(const std::vector<std::string> &names,
                     const std::vector<double> &positions,
                     const std::vector<double> &velocities, double stamp,
                     double now, Feedback &output);
bool safe_target(const Upper &candidate, const Upper &previous,
                 const Feedback &fb, const Config &c);

enum class CalibrationState {
  Origin,
  Forward,
  ReturnForward,
  Left,
  ReturnReady,
  Ready,
  Failed
};
class AxisCalibration {
public:
  explicit AxisCalibration(const Config &config) : config_(config) {}
  void reset();
  bool update(const HandSample &left, const HandSample &right, double now);
  CalibrationState state() const { return state_; }
  Eigen::Matrix3d mapping() const { return mapping_; }
  const char *prompt() const;

private:
  bool stable(const Eigen::Vector3d &l, const Eigen::Vector3d &r, double now);
  bool motion_valid(const Eigen::Vector3d &l, const Eigen::Vector3d &r) const;
  Config config_;
  CalibrationState state_{CalibrationState::Origin};
  Eigen::Matrix3d mapping_{nominal_mapping()};
  Eigen::Vector3d origin_l_{Eigen::Vector3d::Zero()},
      origin_r_{Eigen::Vector3d::Zero()}, forward_{Eigen::Vector3d::Zero()};
  Eigen::Vector3d stable_l_{Eigen::Vector3d::Zero()},
      stable_r_{Eigen::Vector3d::Zero()};
  Eigen::Vector3d sum_l_{Eigen::Vector3d::Zero()},
      sum_r_{Eigen::Vector3d::Zero()};
  double stable_since_{-1};
  size_t count_{0};
  uint64_t last_l_{0}, last_r_{0};
};
struct SolveResult {
  Upper q{Upper::Zero()};
  std::array<double, 2> position_error{}, orientation_error{};
  double elapsed_ms{0};
  bool valid{false}, deadline_missed{false};
};
class Kinematics {
public:
  Kinematics(const std::string &model, const std::string &srdf,
             const Config &config);
  Pose fk(const Arm &q, bool left);
  SolveResult solve(const std::array<Pose, 2> &targets, const Upper &seed);

private:
  fa_arm_kinematic::IKSolver solver_;
  Config config_;
};
std::string verify_bundle(const std::string &root);
const char *release_id();
std::string deployment_id(const Config &config);
std::string file_sha256(const std::string &file);
} // namespace fa
