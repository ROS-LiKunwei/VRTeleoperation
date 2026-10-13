#include "min_snap/min_snap_planner.hpp"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <limits>
#include <sstream>
#include <utility>

namespace min_snap
{

namespace
{
constexpr double kMinSnapMaxSDot = 2.1875;
constexpr double kMinSnapMaxAbsSDDot = 7.513188404399293;
}

template<std::size_t JointCount>
MinSnapPlanner<JointCount>::MinSnapPlanner(std::string name)
: name_(std::move(name))
{
}

template<std::size_t JointCount>
bool MinSnapPlanner<JointCount>::plan(
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
  double * final_duration,
  bool * duration_extended)
{
  last_failure_reason_.clear();
  if (!std::isfinite(max_velocity) || max_velocity <= 0.0 ||
    !std::isfinite(max_acceleration) || max_acceleration <= 0.0)
  {
    last_failure_reason_ = "invalid_constraints max_velocity=" + std::to_string(max_velocity) +
      " max_acceleration=" + std::to_string(max_acceleration);
    return false;
  }

  const double requested_duration = std::max(expected_duration, config.min_duration_s);
  double duration = constrained_duration(
    start_pos, goal_pos, requested_duration, max_velocity, max_acceleration, config.min_duration_s);
  duration = std::max(duration, 1e-4);
  const double max_duration_on_violation =
    std::isfinite(config.max_duration_on_violation_s) && config.max_duration_on_violation_s > 0.0 ?
    config.max_duration_on_violation_s : std::numeric_limits<double>::infinity();
  auto duration_over_limit = [max_duration_on_violation](double value) {
      return std::isfinite(max_duration_on_violation) && value > max_duration_on_violation;
    };
  if (duration_over_limit(duration)) {
    last_failure_reason_ = "duration_limit_exceeded duration=" + std::to_string(duration) +
      " max_duration_on_violation_s=" + std::to_string(max_duration_on_violation);
    if (final_duration != nullptr) {
      *final_duration = duration;
    }
    return false;
  }

  bool ok = false;
  bool extended = duration > requested_duration + 1e-9;
  bool any_solved = false;
  bool duration_limited = false;
  double last_checked_duration = duration;
  const int max_iterations = std::max(1, config.max_duration_search_iterations);
  for (int iter = 0; iter < max_iterations; ++iter) {
    bool solved = true;
    for (std::size_t i = 0; i < JointCount; ++i) {
      solved = trajectories_[i].solve(
        start_pos[i], start_vel[i], start_acc[i], goal_pos[i], 0.0, 0.0, duration,
        config.snap_weight_lambda) && solved;
    }
    if (!solved) {
      duration *= std::max(1.01, config.duration_scale_on_violation);
      extended = true;
      if (duration_over_limit(duration)) {
        duration_limited = true;
        break;
      }
      continue;
    }
    any_solved = true;
    duration_s_ = duration;
    last_checked_duration = duration;
    if (satisfies_constraints(
        max_velocity, max_acceleration,
        static_cast<std::size_t>(std::max(2, config.constraint_sample_count))))
    {
      ok = true;
      break;
    }
    duration *= std::max(1.01, config.duration_scale_on_violation);
    extended = true;
    if (duration_over_limit(duration)) {
      duration_limited = true;
      break;
    }
  }
  if (!ok) {
    last_failure_reason_ = std::string(
      duration_limited ? "duration_limit_exceeded" :
      (any_solved ? "constraint_search_exhausted" : "polynomial_solve_failed")) +
      " duration=" + std::to_string(any_solved ? last_checked_duration : duration) +
      " max_duration_on_violation_s=" + std::to_string(max_duration_on_violation) +
      " iterations=" + std::to_string(max_iterations);
    if (any_solved) {
      last_failure_reason_ += " " + constraint_failure_summary(
          max_velocity, max_acceleration, lower_limit, upper_limit,
          static_cast<std::size_t>(std::max(2, config.constraint_sample_count)),
          last_checked_duration);
    }
    if (final_duration != nullptr) {
      *final_duration = any_solved ? last_checked_duration : duration;
    }
    return false;
  }

  goal_pos_ = goal_pos;
  duration_s_ = duration;
  active_ = true;
  if (final_duration != nullptr) {
    *final_duration = duration_s_;
  }
  if (duration_extended != nullptr) {
    *duration_extended = extended;
  }
  return true;
}

template<std::size_t JointCount>
JointTrajectorySample<JointCount> MinSnapPlanner<JointCount>::sample(double elapsed_time) const
{
  JointTrajectorySample<JointCount> sample;
  const double t = std::clamp(elapsed_time, 0.0, duration_s_);
  for (std::size_t i = 0; i < JointCount; ++i) {
    const auto state = trajectories_[i].sample(t);
    sample.position[i] = state.position;
    sample.velocity[i] = state.velocity;
    sample.acceleration[i] = state.acceleration;
    sample.jerk[i] = state.jerk;
  }
  return sample;
}

template<std::size_t JointCount>
bool MinSnapPlanner<JointCount>::finished(double elapsed_time) const
{
  return active_ && elapsed_time >= duration_s_;
}

template<std::size_t JointCount>
double MinSnapPlanner<JointCount>::constrained_duration(
  const JointArray & start_pos,
  const JointArray & goal_pos,
  double expected_duration,
  double max_velocity,
  double max_acceleration,
  double min_duration) const
{
  double duration = std::max(expected_duration, min_duration);
  for (std::size_t i = 0; i < JointCount; ++i) {
    const double delta = std::abs(goal_pos[i] - start_pos[i]);
    duration = std::max(duration, delta * kMinSnapMaxSDot / max_velocity);
    duration = std::max(duration, std::sqrt(delta * kMinSnapMaxAbsSDDot / max_acceleration));
  }
  return std::max(duration, min_duration);
}

template<std::size_t JointCount>
bool MinSnapPlanner<JointCount>::satisfies_constraints(
  double max_velocity,
  double max_acceleration,
  std::size_t sample_count) const
{
  constexpr double tolerance = 1.000001;
  sample_count = std::max<std::size_t>(2, sample_count);
  for (std::size_t joint = 0; joint < JointCount; ++joint) {
    const auto & trajectory = trajectories_[joint];
    for (std::size_t sample = 0; sample < sample_count; ++sample) {
      const double t = duration_s_ * static_cast<double>(sample) / static_cast<double>(sample_count - 1);
      const auto state = trajectory.sample(t);
      if (std::abs(state.velocity) > max_velocity * tolerance) {
        return false;
      }
      if (std::abs(state.acceleration) > max_acceleration * tolerance) {
        return false;
      }
    }
  }
  return true;
}

template<std::size_t JointCount>
std::string MinSnapPlanner<JointCount>::constraint_failure_summary(
  double max_velocity,
  double max_acceleration,
  const JointArray & lower_limit,
  const JointArray & upper_limit,
  std::size_t sample_count,
  double duration) const
{
  sample_count = std::max<std::size_t>(2, sample_count);
  double peak_velocity = 0.0;
  double peak_acceleration = 0.0;
  double max_limit_violation = 0.0;
  std::size_t peak_velocity_joint = 0;
  std::size_t peak_acceleration_joint = 0;
  std::size_t limit_violation_joint = 0;
  double peak_velocity_time = 0.0;
  double peak_acceleration_time = 0.0;
  for (std::size_t joint = 0; joint < JointCount; ++joint) {
    const auto & trajectory = trajectories_[joint];
    for (std::size_t sample = 0; sample < sample_count; ++sample) {
      const double t = duration * static_cast<double>(sample) / static_cast<double>(sample_count - 1);
      const auto state = trajectory.sample(t);
      const double velocity = std::abs(state.velocity);
      const double acceleration = std::abs(state.acceleration);
      if (velocity > peak_velocity) {
        peak_velocity = velocity;
        peak_velocity_joint = joint;
        peak_velocity_time = t;
      }
      if (acceleration > peak_acceleration) {
        peak_acceleration = acceleration;
        peak_acceleration_joint = joint;
        peak_acceleration_time = t;
      }
      const double lower_violation = lower_limit[joint] - state.position;
      const double upper_violation = state.position - upper_limit[joint];
      const double violation = std::max(lower_violation, upper_violation);
      if (violation > max_limit_violation) {
        max_limit_violation = violation;
        limit_violation_joint = joint;
      }
    }
  }
  std::ostringstream stream;
  stream << std::fixed << std::setprecision(6)
         << "peak_velocity=" << peak_velocity
         << " peak_velocity_joint=" << peak_velocity_joint
         << " peak_velocity_time=" << peak_velocity_time
         << " max_velocity=" << max_velocity
         << " peak_acceleration=" << peak_acceleration
         << " peak_acceleration_joint=" << peak_acceleration_joint
         << " peak_acceleration_time=" << peak_acceleration_time
         << " max_acceleration=" << max_acceleration
         << " max_limit_violation=" << max_limit_violation
         << " limit_violation_joint=" << limit_violation_joint;
  return stream.str();
}

template class MinSnapPlanner<kArmJointCount>;
template class MinSnapPlanner<kNeckJointCount>;

}  // namespace min_snap
