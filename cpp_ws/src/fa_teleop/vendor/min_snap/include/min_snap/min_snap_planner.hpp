#pragma once

#include <array>
#include <cstddef>
#include <string>

#include "min_snap/fa_joint_mapping.hpp"
#include "min_snap/min_snap_trajectory.hpp"

namespace min_snap
{

template<std::size_t JointCount>
struct JointTrajectorySample
{
  std::array<double, JointCount> position{};
  std::array<double, JointCount> velocity{};
  std::array<double, JointCount> acceleration{};
  std::array<double, JointCount> jerk{};
};

using ArmTrajectorySample = JointTrajectorySample<kArmJointCount>;
using NeckTrajectorySample = JointTrajectorySample<kNeckJointCount>;

struct PlannerConfig
{
  double min_duration_s{0.01};
  double duration_scale_on_violation{1.1};
  double max_duration_on_violation_s{2.0};
  int max_duration_search_iterations{100};
  int constraint_sample_count{500};
  double snap_weight_lambda{0.85};
};

template<std::size_t JointCount>
class MinSnapPlanner
{
public:
  using JointArray = std::array<double, JointCount>;

  explicit MinSnapPlanner(std::string name = "joint_group");

  bool plan(
    const JointArray & start_pos,
    const JointArray & start_vel,
    const JointArray & start_acc,
    const JointArray & goal_pos,
    double expected_duration,
    double max_velocity,
    double max_acceleration,
    const JointArray & lower_limit,
    const JointArray & upper_limit,
    const PlannerConfig & config,
    double * final_duration = nullptr,
    bool * duration_extended = nullptr);

  JointTrajectorySample<JointCount> sample(double elapsed_time) const;
  bool active() const { return active_; }
  void deactivate() { active_ = false; }
  bool finished(double elapsed_time) const;
  double duration() const { return duration_s_; }
  const JointArray & goal() const { return goal_pos_; }
  const std::string & last_failure_reason() const { return last_failure_reason_; }

private:
  double constrained_duration(
    const JointArray & start_pos,
    const JointArray & goal_pos,
    double expected_duration,
    double max_velocity,
    double max_acceleration,
    double min_duration) const;

  bool satisfies_constraints(
    double max_velocity,
    double max_acceleration,
    std::size_t sample_count) const;

  std::string constraint_failure_summary(
    double max_velocity,
    double max_acceleration,
    const JointArray & lower_limit,
    const JointArray & upper_limit,
    std::size_t sample_count,
    double duration) const;

  std::string name_;
  std::array<MinSnapTrajectory, JointCount> trajectories_{};
  JointArray goal_pos_{};
  std::string last_failure_reason_;
  double duration_s_{0.0};
  bool active_{false};
};

using MinSnapArmPlanner = MinSnapPlanner<kArmJointCount>;
using MinSnapNeckPlanner = MinSnapPlanner<kNeckJointCount>;

}  // namespace min_snap
