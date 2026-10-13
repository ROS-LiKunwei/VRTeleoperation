#include <algorithm>
#include <array>
#include <chrono>
#include <cctype>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <memory>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

#include "min_snap/fa_joint_mapping.hpp"
#include "min_snap/min_snap_planner.hpp"
#include "min_snap/msg/min_snap_target.hpp"
#include "min_snap/trajectory_recorder.hpp"
#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/joint_state.hpp"
#include "std_msgs/msg/float64_multi_array.hpp"
#include "std_srvs/srv/trigger.hpp"

namespace min_snap
{

namespace
{

bool finite_vector(const std::vector<double> & values)
{
  return std::all_of(values.begin(), values.end(), [](double value) { return std::isfinite(value); });
}

ArmArray to_arm_array(const std::vector<double> & values)
{
  ArmArray out{};
  std::copy_n(values.begin(), std::min(values.size(), out.size()), out.begin());
  return out;
}

NeckArray to_neck_array(const std::vector<double> & values)
{
  NeckArray out{};
  std::copy_n(values.begin(), std::min(values.size(), out.size()), out.begin());
  return out;
}

std::string format_double(double value)
{
  if (!std::isfinite(value)) {
    return "nan";
  }
  std::ostringstream stream;
  stream << std::fixed << std::setprecision(6) << value;
  return stream.str();
}

std::string format_arm(const ArmArray & values, std::size_t count)
{
  std::ostringstream stream;
  stream << "[";
  for (std::size_t i = 0; i < count; ++i) {
    if (i > 0) {
      stream << ",";
    }
    stream << format_double(values[i]);
  }
  stream << "]";
  return stream.str();
}

std::string format_arm_actual(
  const std::unordered_map<std::string, double> & values,
  const std::array<std::string, kArmJointCount> & names,
  std::size_t count)
{
  std::ostringstream stream;
  stream << "[";
  for (std::size_t i = 0; i < count; ++i) {
    if (i > 0) {
      stream << ",";
    }
    const auto it = values.find(names[i]);
    stream << (it == values.end() ? "nan" : format_double(it->second));
  }
  stream << "]";
  return stream.str();
}

std::string format_neck(const NeckArray & values)
{
  std::ostringstream stream;
  stream << "[";
  for (std::size_t i = 0; i < kNeckJointCount; ++i) {
    if (i > 0) {
      stream << ",";
    }
    stream << format_double(values[i]);
  }
  stream << "]";
  return stream.str();
}

std::string format_neck_actual(
  const std::unordered_map<std::string, double> & values,
  const std::array<std::string, kNeckJointCount> & names)
{
  std::ostringstream stream;
  stream << "[";
  for (std::size_t i = 0; i < kNeckJointCount; ++i) {
    if (i > 0) {
      stream << ",";
    }
    const auto it = values.find(names[i]);
    stream << (it == values.end() ? "nan" : format_double(it->second));
  }
  stream << "]";
  return stream.str();
}

double max_abs_error(
  const std::unordered_map<std::string, double> & actual,
  const std::array<std::string, kArmJointCount> & names,
  const ArmArray & desired,
  std::size_t count)
{
  double out = 0.0;
  for (std::size_t i = 0; i < count; ++i) {
    const auto it = actual.find(names[i]);
    if (it != actual.end() && std::isfinite(it->second)) {
      out = std::max(out, std::abs(it->second - desired[i]));
    }
  }
  return out;
}

double max_abs_neck_error(
  const NeckArray & actual,
  const NeckArray & desired)
{
  double out = 0.0;
  for (std::size_t i = 0; i < kNeckJointCount; ++i) {
    if (std::isfinite(actual[i])) {
      out = std::max(out, std::abs(actual[i] - desired[i]));
    }
  }
  return out;
}

ArmArray make_arm_array(const std::array<double, kArmJointCount> & values)
{
  ArmArray out{};
  std::copy(values.begin(), values.end(), out.begin());
  return out;
}

NeckArray make_neck_array(const std::array<double, kNeckJointCount> & values)
{
  NeckArray out{};
  std::copy(values.begin(), values.end(), out.begin());
  return out;
}

}  // namespace

class MinSnapNode final : public rclcpp::Node
{
public:
  MinSnapNode()
  : Node("min_snap_node"),
    left_planner_("left"),
    right_planner_("right"),
    neck_planner_("neck")
  {
    load_parameters();
    open_run_log();

    command_pub_ = create_publisher<std_msgs::msg::Float64MultiArray>(command_topic_, 10);
    desired_joint_state_pub_ = create_publisher<sensor_msgs::msg::JointState>(desired_joint_states_topic_, 10);
    initialize_reusable_messages();
    target_sub_ = create_subscription<min_snap::msg::MinSnapTarget>(
      target_topic_, rclcpp::QoS(rclcpp::KeepLast(1)),
      [this](const min_snap::msg::MinSnapTarget::SharedPtr msg) { on_target(*msg); });
    joint_state_sub_ = create_subscription<sensor_msgs::msg::JointState>(
      joint_states_topic_, 20,
      [this](const sensor_msgs::msg::JointState::SharedPtr msg) { on_joint_state(*msg); });
    pause_publish_service_ = create_service<std_srvs::srv::Trigger>(
      pause_publish_service_name_,
      [this](
        const std::shared_ptr<std_srvs::srv::Trigger::Request> request,
        std::shared_ptr<std_srvs::srv::Trigger::Response> response) {
        on_pause_publish_request(request, response);
      });
    resume_publish_service_ = create_service<std_srvs::srv::Trigger>(
      resume_publish_service_name_,
      [this](
        const std::shared_ptr<std_srvs::srv::Trigger::Request> request,
        std::shared_ptr<std_srvs::srv::Trigger::Response> response) {
        on_resume_publish_request(request, response);
      });

    if (record_tracking_) {
      if (recorder_.open(tracking_output_dir_)) {
        RCLCPP_INFO(get_logger(), "Min-snap tracking CSV: %s", recorder_.path().c_str());
      } else {
        RCLCPP_ERROR(get_logger(), "Failed to open tracking CSV directory: %s", tracking_output_dir_.c_str());
      }
    }

    const auto period = std::chrono::duration<double>(1.0 / std::max(1.0, publish_hz_));
    timer_ = create_wall_timer(
      std::chrono::duration_cast<std::chrono::nanoseconds>(period),
      [this]() { on_timer(); });
    timer_->cancel();

    RCLCPP_INFO(
      get_logger(),
      "min_snap_node ready: target=%s command=%s desired_joint_states=%s joint_states=%s publish_hz=%.3f",
      target_topic_.c_str(), command_topic_.c_str(), desired_joint_states_topic_.c_str(),
      joint_states_topic_.c_str(), publish_hz_);
    write_run_log(
      "node_ready robot_type=" + robot_type_ +
      " target_topic=" + target_topic_ +
      " command_topic=" + command_topic_ +
      " joint_states_topic=" + joint_states_topic_ +
      " pause_publish_service=" + pause_publish_service_name_ +
      " resume_publish_service=" + resume_publish_service_name_);
  }

  ~MinSnapNode() override
  {
    if (run_log_file_.is_open()) {
      write_run_log("node_shutdown");
      run_log_file_.flush();
      run_log_file_.close();
    }
  }

private:
  void load_parameters()
  {
    robot_type_ = declare_parameter<std::string>("robot_type", "fa");
    std::transform(robot_type_.begin(), robot_type_.end(), robot_type_.begin(), ::tolower);
    if (robot_type_ == "sysmo" || robot_type_ == "sysmo-32") {
      robot_type_ = "sysmo32";
    }
    if (robot_type_ != "fa" && robot_type_ != "sysmo32") {
      RCLCPP_WARN(get_logger(), "Unsupported robot_type '%s'; falling back to 'fa'", robot_type_.c_str());
      robot_type_ = "fa";
    }
    active_arm_joint_count_ = robot_type_ == "sysmo32" ? kSysmo32ArmJointCount : kArmJointCount;
    if (robot_type_ == "sysmo32") {
      left_lower_limits_.fill(-M_PI);
      left_upper_limits_.fill(M_PI);
      right_lower_limits_.fill(-M_PI);
      right_upper_limits_.fill(M_PI);
    }
    publish_hz_ = declare_parameter<double>("publish_hz", 1000.0);
    target_topic_ = declare_parameter<std::string>("target_topic", "/min_snap/target");
    const std::string default_command_topic =
      robot_type_ == "sysmo32" ? "/sysmo_left_arm_controller/commands" : "/upper_position_controller/commands";
    command_topic_ = declare_parameter<std::string>("command_topic", default_command_topic);
    desired_joint_states_topic_ =
      declare_parameter<std::string>("desired_joint_states_topic", "/min_snap/desired_joint_states");
    joint_states_topic_ = declare_parameter<std::string>("joint_states_topic", "/joint_states");
    pause_publish_service_name_ =
      declare_parameter<std::string>("pause_publish_service", "/min_snap/pause_trajectory_publish");
    resume_publish_service_name_ =
      declare_parameter<std::string>("resume_publish_service", "/min_snap/resume_trajectory_publish");
    planner_config_.min_duration_s = declare_parameter<double>("min_duration_s", 0.01);
    default_expected_duration_s_ = declare_parameter<double>("default_expected_duration_s", 0.016);
    default_max_velocity_rad_s_ = declare_parameter<double>("default_max_velocity_rad_s", 1.5);
    default_max_acceleration_rad_s2_ = declare_parameter<double>("default_max_acceleration_rad_s2", 15.0);
    planner_config_.duration_scale_on_violation =
      declare_parameter<double>("duration_scale_on_violation", 1.1);
    planner_config_.max_duration_on_violation_s =
      declare_parameter<double>("max_duration_on_violation_s", 2.0);
    planner_config_.max_duration_search_iterations =
      declare_parameter<int>("max_duration_search_iterations", 100);
    planner_config_.constraint_sample_count = declare_parameter<int>("constraint_sample_count", 500);
    planner_config_.snap_weight_lambda = declare_parameter<double>("snap_weight_lambda", 0.85);
    replan_threshold_rad_ = declare_parameter<double>("replan_threshold_rad", 0.0005);
    publish_position_abs_limit_rad_ = declare_parameter<double>("publish_position_abs_limit_rad", 4.0);
    publish_joint_limit_margin_rad_ = declare_parameter<double>("publish_joint_limit_margin_rad", 0.02);
    publish_velocity_abs_limit_rad_s_ = declare_parameter<double>("publish_velocity_abs_limit_rad_s", 2.0);
    publish_acceleration_abs_limit_rad_s2_ = declare_parameter<double>("publish_acceleration_abs_limit_rad_s2", 20.0);
    use_joint_state_velocity_ = declare_parameter<bool>("use_joint_state_velocity", true);
    joint_state_velocity_abs_limit_rad_s_ =
      declare_parameter<double>("joint_state_velocity_abs_limit_rad_s", 5.0);
    replan_from_joint_state_ = declare_parameter<bool>("replan_from_joint_state", false);
    min_replan_interval_s_ = declare_parameter<double>("min_replan_interval_s", 0.016);
    require_joint_state_before_start_ = declare_parameter<bool>("require_joint_state_before_start", true);
    joint_state_timeout_s_ = declare_parameter<double>("joint_state_timeout_s", 0.5);
    goal_reached_position_tolerance_rad_ =
      declare_parameter<double>("goal_reached_position_tolerance_rad", 0.01);
    goal_reached_hold_s_ = declare_parameter<double>("goal_reached_hold_s", 0.05);
    record_tracking_ = declare_parameter<bool>("record_tracking", false);
    tracking_output_dir_ = declare_parameter<std::string>(
      "tracking_output_dir", "/home/likunwei/humanoid_ws/src/min_snap/logs");
    record_run_log_ = declare_parameter<bool>("record_run_log", true);
    run_log_output_dir_ = declare_parameter<std::string>(
      "run_log_output_dir", "/home/likunwei/humanoid_ws/src/min_snap/Log");
    run_log_period_s_ = declare_parameter<double>("run_log_period_s", 0.1);
    run_log_warn_throttle_s_ = declare_parameter<double>("run_log_warn_throttle_s", 2.0);
    const auto neck_default = declare_parameter<std::vector<double>>("neck_default_position", {0.0, 0.0});
    if (neck_default.size() == kNeckJointCount && finite_vector(neck_default)) {
      neck_default_[0] = neck_default[0];
      neck_default_[1] = neck_default[1];
    }
    speed_mode_ = declare_parameter<double>("speed_mode", 4.0);
    neck_joint_ = declare_parameter<double>("neck_joint", 0.0);
    const auto reserved = declare_parameter<std::vector<double>>("reserved", {0.0, 0.0, 0.0, 0.0});
    if (reserved.size() == kSysmo32ReservedCount && finite_vector(reserved)) {
      std::copy(reserved.begin(), reserved.end(), reserved_.begin());
    }
  }

  void on_joint_state(const sensor_msgs::msg::JointState & msg)
  {
    std::array<bool, kArmJointCount> left_seen{};
    std::array<bool, kArmJointCount> right_seen{};
    std::array<bool, kNeckJointCount> neck_seen{};
    latest_left_velocities_ = ArmArray{};
    latest_right_velocities_ = ArmArray{};
    latest_neck_velocities_ = NeckArray{};
    if (recorder_.enabled()) {
      latest_positions_.clear();
      latest_velocities_.clear();
    }
    const bool velocity_size_matches_names = msg.velocity.size() >= msg.name.size();
    for (std::size_t i = 0; i < msg.name.size(); ++i) {
      const auto slot_it = joint_state_name_slots_.find(msg.name[i]);
      const bool known_joint = slot_it != joint_state_name_slots_.end();
      if (i < msg.position.size()) {
        if (known_joint) {
          set_cached_joint_position(slot_it->second, msg.position[i], left_seen, right_seen, neck_seen);
        }
        if (recorder_.enabled()) {
          latest_positions_[msg.name[i]] = msg.position[i];
        }
      }
      if (use_joint_state_velocity_ && velocity_size_matches_names) {
        const double velocity = msg.velocity[i];
        if (joint_state_velocity_valid(velocity)) {
          if (known_joint) {
            set_cached_joint_velocity(slot_it->second, velocity);
          }
          if (recorder_.enabled()) {
            latest_velocities_[msg.name[i]] = velocity;
          }
        } else {
          RCLCPP_WARN_THROTTLE(
            get_logger(), *get_clock(), 2000,
            "Ignoring invalid /joint_states velocity for joint '%s': %.6f limit %.6f",
            msg.name[i].c_str(), velocity, joint_state_velocity_abs_limit_rad_s_);
          write_run_log_warn_throttled(
            "joint_state_velocity_invalid:" + msg.name[i],
            "joint_state_velocity_ignored joint=" + msg.name[i] +
            " value=" + format_double(velocity) +
            " limit=" + format_double(joint_state_velocity_abs_limit_rad_s_));
        }
      }
    }
    if (use_joint_state_velocity_ && !velocity_size_matches_names) {
      RCLCPP_WARN_THROTTLE(
        get_logger(), *get_clock(), 2000,
        "Ignoring /joint_states velocity: velocity size %zu is smaller than name size %zu",
        msg.velocity.size(), msg.name.size());
      write_run_log_warn_throttled(
        "joint_state_velocity_size_mismatch",
        "joint_state_velocity_ignored reason=size_mismatch velocity_size=" +
        std::to_string(msg.velocity.size()) + " name_size=" + std::to_string(msg.name.size()));
    }
    last_joint_state_time_ = now();

    has_arm_joint_state_ = true;
    for (std::size_t i = 0; i < active_arm_joint_count_; ++i) {
      if (!left_seen[i] || !right_seen[i]) {
        has_arm_joint_state_ = false;
        break;
      }
    }
    has_neck_joint_state_ = true;
    for (std::size_t i = 0; i < kNeckJointCount; ++i) {
      if (!neck_seen[i]) {
        has_neck_joint_state_ = false;
        break;
      }
    }
    if (!has_arm_joint_state_) {
      RCLCPP_WARN_THROTTLE(
        get_logger(), *get_clock(), 2000,
        "/joint_states missing required %s arm joints; waiting for complete joint state",
        robot_type_.c_str());
    }
  }

  bool joint_state_velocity_valid(double velocity) const
  {
    if (!std::isfinite(velocity)) {
      return false;
    }
    if (joint_state_velocity_abs_limit_rad_s_ <= 0.0 ||
      !std::isfinite(joint_state_velocity_abs_limit_rad_s_))
    {
      return true;
    }
    return std::abs(velocity) <= joint_state_velocity_abs_limit_rad_s_;
  }

  void on_target(const min_snap::msg::MinSnapTarget & msg)
  {
    if (!validate_target(msg)) {
      return;
    }
    if (require_joint_state_before_start_ && !joint_state_fresh()) {
      RCLCPP_WARN(get_logger(), "Rejecting /min_snap/target: /joint_states is not fresh");
      write_run_log("target_rejected reason=joint_states_not_fresh");
      return;
    }

    const bool left_target_provided = !msg.left_arm_target_rad.empty();
    const bool right_target_provided = !msg.right_arm_target_rad.empty();
    const bool neck_target_provided = !msg.neck_target_rad.empty();
    ArmArray left_goal = left_target_provided ? to_arm_array(msg.left_arm_target_rad) : latest_left_positions_;
    ArmArray right_goal = right_target_provided ? to_arm_array(msg.right_arm_target_rad) : latest_right_positions_;
    NeckArray neck_goal = neck_target_provided ? to_neck_array(msg.neck_target_rad) : latest_neck_position_;
    clamp_goal_to_joint_limits(left_goal, "left");
    clamp_goal_to_joint_limits(right_goal, "right");
    clamp_goal_to_neck_limits(neck_goal);
    ++target_sequence_;
    if (target_delta_below_threshold(left_goal, right_goal, neck_goal)) {
      return;
    }

    ArmArray start_left_pos{};
    ArmArray start_left_vel{};
    ArmArray start_left_acc{};
    ArmArray start_right_pos{};
    ArmArray start_right_vel{};
    ArmArray start_right_acc{};
    NeckArray start_neck_pos{};
    NeckArray start_neck_vel{};
    NeckArray start_neck_acc{};

    const auto now_time = now();
    const bool replanning = left_planner_.active() || right_planner_.active() || neck_planner_.active();
    if (replanning && last_plan_time_.nanoseconds() != 0 &&
      (now_time - last_plan_time_).seconds() < std::max(0.0, min_replan_interval_s_))
    {
      return;
    }

    if (replan_from_joint_state_ || !replanning) {
      if (!start_from_joint_state(
          start_left_pos, start_left_vel, start_right_pos, start_right_vel, start_neck_pos, start_neck_vel))
      {
        RCLCPP_WARN(get_logger(), "Rejecting /min_snap/target: cannot build start state from /joint_states");
        write_run_log("target_rejected seq=" + std::to_string(target_sequence_) + " reason=cannot_build_start_state");
        return;
      }
    } else {
      const double elapsed = (now_time - trajectory_start_time_).seconds();
      auto left_sample = left_planner_.sample(elapsed);
      auto right_sample = right_planner_.sample(elapsed);
      auto neck_sample = neck_planner_.active() ? neck_planner_.sample(elapsed) : NeckTrajectorySample{};
      if (!clamp_sample_to_publish_limits(left_sample, "left_replan_start") ||
        !clamp_sample_to_publish_limits(right_sample, "right_replan_start") ||
        (neck_planning_enabled() && !clamp_neck_sample_to_publish_limits(neck_sample, "neck_replan_start")))
      {
        deactivate_trajectories("invalid_replan_start_state");
        return;
      }
      start_left_pos = left_sample.position;
      start_left_vel = left_sample.velocity;
      start_left_acc = left_sample.acceleration;
      start_right_pos = right_sample.position;
      start_right_vel = right_sample.velocity;
      start_right_acc = right_sample.acceleration;
      start_neck_pos = neck_planner_.active() ? neck_sample.position : latest_neck_position_;
      start_neck_vel = neck_planner_.active() ? neck_sample.velocity : latest_neck_velocities_;
      start_neck_acc = neck_planner_.active() ? neck_sample.acceleration : NeckArray{};
    }
    if (!left_target_provided) {
      start_left_pos = latest_left_positions_;
      start_left_vel = use_joint_state_velocity_ ? latest_left_velocities_ : ArmArray{};
      start_left_acc = ArmArray{};
    }
    if (!right_target_provided) {
      start_right_pos = latest_right_positions_;
      start_right_vel = use_joint_state_velocity_ ? latest_right_velocities_ : ArmArray{};
      start_right_acc = ArmArray{};
    }
    if (!neck_target_provided) {
      start_neck_pos = latest_neck_position_;
      start_neck_vel = use_joint_state_velocity_ ? latest_neck_velocities_ : NeckArray{};
      start_neck_acc = NeckArray{};
    }

    const double expected_duration =
      msg.expected_duration_s > 0.0 ? msg.expected_duration_s : default_expected_duration_s_;
    const double max_velocity =
      msg.max_velocity_rad_s > 0.0 ? msg.max_velocity_rad_s : default_max_velocity_rad_s_;
    const double max_acceleration =
      msg.max_acceleration_rad_s2 > 0.0 ? msg.max_acceleration_rad_s2 : default_max_acceleration_rad_s2_;
    clamp_start_velocity_to_limit(start_left_vel, max_velocity, "left");
    clamp_start_velocity_to_limit(start_right_vel, max_velocity, "right");
    clamp_neck_start_velocity_to_limit(start_neck_vel, max_velocity);

    double left_duration = 0.0;
    double right_duration = 0.0;
    double neck_duration = 0.0;
    bool left_extended = false;
    bool right_extended = false;
    bool neck_extended = false;
    const ArmArray left_lower = soft_lower_limits(left_lower_limits_, left_upper_limits_);
    const ArmArray left_upper = soft_upper_limits(left_lower_limits_, left_upper_limits_);
    const ArmArray right_lower = soft_lower_limits(right_lower_limits_, right_upper_limits_);
    const ArmArray right_upper = soft_upper_limits(right_lower_limits_, right_upper_limits_);
    const NeckArray neck_lower = soft_neck_lower_limits();
    const NeckArray neck_upper = soft_neck_upper_limits();
    const auto plan_start_time = std::chrono::steady_clock::now();
    const auto left_plan_start_time = std::chrono::steady_clock::now();
    const bool left_ok = left_planner_.plan(
      start_left_pos, start_left_vel, start_left_acc, left_goal, expected_duration,
      max_velocity, max_acceleration, left_lower, left_upper, planner_config_, &left_duration, &left_extended);
    const auto right_plan_start_time = std::chrono::steady_clock::now();
    const bool right_ok = right_planner_.plan(
      start_right_pos, start_right_vel, start_right_acc, right_goal, expected_duration,
      max_velocity, max_acceleration, right_lower, right_upper, planner_config_, &right_duration, &right_extended);
    const auto neck_plan_start_time = std::chrono::steady_clock::now();
    const bool neck_ok = !neck_planning_enabled() || neck_planner_.plan(
      start_neck_pos, start_neck_vel, start_neck_acc, neck_goal, expected_duration,
      max_velocity, max_acceleration, neck_lower, neck_upper, planner_config_, &neck_duration, &neck_extended);
    const auto plan_end_time = std::chrono::steady_clock::now();
    const double left_plan_ms = std::chrono::duration<double, std::milli>(
      right_plan_start_time - left_plan_start_time).count();
    const double right_plan_ms = std::chrono::duration<double, std::milli>(
      neck_plan_start_time - right_plan_start_time).count();
    const double neck_plan_ms = std::chrono::duration<double, std::milli>(
      plan_end_time - neck_plan_start_time).count();
    const double total_plan_ms = std::chrono::duration<double, std::milli>(
      plan_end_time - plan_start_time).count();

    if (!left_ok || !right_ok || !neck_ok) {
      RCLCPP_ERROR(
        get_logger(),
        "Failed to generate min-snap trajectory: plan_time_ms total=%.3f left=%.3f right=%.3f neck=%.3f",
        total_plan_ms, left_plan_ms, right_plan_ms, neck_plan_ms);
      write_run_log(
        "Failed to generate min-snap trajectory: seq=" + std::to_string(target_sequence_) +
        " plan_time_ms_total=" + format_double(total_plan_ms) +
        " plan_time_ms_left=" + format_double(left_plan_ms) +
        " plan_time_ms_right=" + format_double(right_plan_ms) +
        " plan_time_ms_neck=" + format_double(neck_plan_ms) +
        " left_ok=" + std::string(left_ok ? "true" : "false") +
        " right_ok=" + std::string(right_ok ? "true" : "false") +
        " neck_ok=" + std::string(neck_ok ? "true" : "false") +
        " left_reason=" + left_planner_.last_failure_reason() +
        " right_reason=" + right_planner_.last_failure_reason() +
        " neck_reason=" + neck_planner_.last_failure_reason() +
        " start_left_pos=" + format_arm(start_left_pos, active_arm_joint_count_) +
        " start_right_pos=" + format_arm(start_right_pos, active_arm_joint_count_) +
        " start_neck_pos=" + format_neck(start_neck_pos) +
        " left_goal=" + format_arm(left_goal, active_arm_joint_count_) +
        " right_goal=" + format_arm(right_goal, active_arm_joint_count_) +
        " neck_goal=" + format_neck(neck_goal),
        "ERROR");
      deactivate_trajectories("plan_failed");
      return;
    }

    trajectory_start_time_ = now_time;
    last_plan_time_ = now_time;
    goal_reached_since_ = rclcpp::Time(0, 0, get_clock()->get_clock_type());
    latest_left_goal_ = left_goal;
    latest_right_goal_ = right_goal;
    latest_neck_goal_ = neck_goal;
    has_goal_ = true;
    if (!trajectory_publish_paused_) {
      timer_->reset();
    }

    const double final_duration = std::max({left_duration, right_duration, neck_duration});
    if (left_extended || right_extended || neck_extended || final_duration > expected_duration + 1e-9) {
      RCLCPP_WARN(
        get_logger(),
        "Requested duration %.3f s violates velocity/acceleration limits. Extended trajectory duration to %.3f s.",
        expected_duration, final_duration);
      write_run_log(
        "Requested duration " + format_double(expected_duration) +
        " s violates velocity/acceleration limits. Extended trajectory duration to " +
        format_double(final_duration) + " s.",
        "WARN");
    }
    write_run_log(
      "plan_timing seq=" + std::to_string(target_sequence_) +
      " total_ms=" + format_double(total_plan_ms) +
      " left_ms=" + format_double(left_plan_ms) +
      " right_ms=" + format_double(right_plan_ms) +
      " neck_ms=" + format_double(neck_plan_ms) +
      " expected_duration_s=" + format_double(expected_duration) +
      " final_duration_s=" + format_double(final_duration) +
      " left_duration_s=" + format_double(left_duration) +
      " right_duration_s=" + format_double(right_duration) +
      " neck_duration_s=" + format_double(neck_duration) +
      " left_extended=" + std::string(left_extended ? "true" : "false") +
      " right_extended=" + std::string(right_extended ? "true" : "false") +
      " neck_extended=" + std::string(neck_extended ? "true" : "false"));
    if (replanning) {
      RCLCPP_INFO(get_logger(), "Replanned min-snap trajectory from active trajectory state.");
      write_run_log("Replanned min-snap trajectory from active trajectory state.");
    } else {
      RCLCPP_INFO(get_logger(), "Planned min-snap trajectory duration %.3f s.", final_duration);
      write_run_log("Planned min-snap trajectory duration " + format_double(final_duration) + " s.");
    }
  }

  bool validate_target(const min_snap::msg::MinSnapTarget & msg)
  {
    const auto arm_target_size_valid = [this](const std::vector<double> & target) {
        return target.empty() ||
        (target.size() >= active_arm_joint_count_ && target.size() <= kArmJointCount);
      };
    if (!arm_target_size_valid(msg.left_arm_target_rad) || !arm_target_size_valid(msg.right_arm_target_rad))
    {
      RCLCPP_ERROR(
        get_logger(), "Rejecting target: left/right target arrays must be empty or have %zu to 7 values",
        active_arm_joint_count_);
      return false;
    }
    if (!finite_vector(msg.left_arm_target_rad) || !finite_vector(msg.right_arm_target_rad)) {
      RCLCPP_ERROR(get_logger(), "Rejecting target: target contains NaN or Inf");
      return false;
    }
    if (!msg.neck_target_rad.empty() && msg.neck_target_rad.size() != kNeckJointCount) {
      RCLCPP_ERROR(
        get_logger(), "Rejecting target: neck_target_rad must be empty or have %zu values",
        kNeckJointCount);
      return false;
    }
    if (!finite_vector(msg.neck_target_rad)) {
      RCLCPP_ERROR(get_logger(), "Rejecting target: neck target contains NaN or Inf");
      return false;
    }
    if (msg.max_velocity_rad_s <= 0.0 || !std::isfinite(msg.max_velocity_rad_s)) {
      RCLCPP_ERROR(get_logger(), "Rejecting target: max_velocity_rad_s must be > 0");
      return false;
    }
    if (msg.max_acceleration_rad_s2 <= 0.0 || !std::isfinite(msg.max_acceleration_rad_s2)) {
      RCLCPP_ERROR(get_logger(), "Rejecting target: max_acceleration_rad_s2 must be > 0");
      return false;
    }
    if (msg.expected_duration_s <= 0.0 || !std::isfinite(msg.expected_duration_s)) {
      RCLCPP_WARN(
        get_logger(), "expected_duration_s %.6f is invalid; using default %.6f",
        msg.expected_duration_s, default_expected_duration_s_);
    }
    return true;
  }

  bool target_delta_below_threshold(
    const ArmArray & left_goal,
    const ArmArray & right_goal,
    const NeckArray & neck_goal) const
  {
    if (!has_goal_) {
      return false;
    }
    double max_delta = 0.0;
    for (std::size_t i = 0; i < active_arm_joint_count_; ++i) {
      max_delta = std::max(max_delta, std::abs(left_goal[i] - latest_left_goal_[i]));
      max_delta = std::max(max_delta, std::abs(right_goal[i] - latest_right_goal_[i]));
    }
    if (neck_planning_enabled()) {
      for (std::size_t i = 0; i < kNeckJointCount; ++i) {
        max_delta = std::max(max_delta, std::abs(neck_goal[i] - latest_neck_goal_[i]));
      }
    }
    return max_delta < replan_threshold_rad_;
  }

  bool start_from_joint_state(
    ArmArray & left_pos,
    ArmArray & left_vel,
    ArmArray & right_pos,
    ArmArray & right_vel,
    NeckArray & neck_pos,
    NeckArray & neck_vel) const
  {
    if (!has_arm_joint_state_) {
      return false;
    }
    left_pos = latest_left_positions_;
    right_pos = latest_right_positions_;
    left_vel = use_joint_state_velocity_ ? latest_left_velocities_ : ArmArray{};
    right_vel = use_joint_state_velocity_ ? latest_right_velocities_ : ArmArray{};
    neck_pos = latest_neck_position_;
    neck_vel = use_joint_state_velocity_ ? latest_neck_velocities_ : NeckArray{};
    return true;
  }

  bool neck_planning_enabled() const
  {
    return robot_type_ == "fa";
  }

  bool extract_active_arm_positions(
    const std::unordered_map<std::string, double> & positions,
    ArmArray & left,
    ArmArray & right) const
  {
    left = ArmArray{};
    right = ArmArray{};
    const auto & left_names = left_arm_joint_names();
    const auto & right_names = right_arm_joint_names();
    for (std::size_t i = 0; i < active_arm_joint_count_; ++i) {
      const auto left_it = positions.find(left_names[i]);
      const auto right_it = positions.find(right_names[i]);
      if (left_it == positions.end() || right_it == positions.end()) {
        return false;
      }
      left[i] = left_it->second;
      right[i] = right_it->second;
    }
    return true;
  }

  bool joint_state_fresh() const
  {
    if (!has_arm_joint_state_) {
      return false;
    }
    if (last_joint_state_time_.nanoseconds() == 0) {
      return false;
    }
    return (now() - last_joint_state_time_).seconds() <= joint_state_timeout_s_;
  }

  void on_pause_publish_request(
    const std::shared_ptr<std_srvs::srv::Trigger::Request> /* request */,
    std::shared_ptr<std_srvs::srv::Trigger::Response> response)
  {
    trajectory_publish_paused_ = true;
    deactivate_trajectories("pause_publish_service");

    const bool publish_stopped =
      trajectory_publish_paused_ && !left_planner_.active() && !right_planner_.active() && !neck_planner_.active();
    response->success = publish_stopped;
    response->message = publish_stopped ?
      "min_snap trajectory point publishing is paused" :
      "failed to pause min_snap trajectory point publishing";

    if (publish_stopped) {
      RCLCPP_INFO(get_logger(), "Paused min-snap trajectory point publishing by service request.");
      write_run_log("pause_publish_service success=true");
    } else {
      RCLCPP_ERROR(get_logger(), "Failed to pause min-snap trajectory point publishing.");
      write_run_log("pause_publish_service success=false", "ERROR");
    }
  }

  void on_resume_publish_request(
    const std::shared_ptr<std_srvs::srv::Trigger::Request> /* request */,
    std::shared_ptr<std_srvs::srv::Trigger::Response> response)
  {
    trajectory_publish_paused_ = false;
    response->success = true;
    response->message = "min_snap trajectory point publishing is resumed";

    RCLCPP_INFO(get_logger(), "Resumed min-snap trajectory point publishing by service request.");
    write_run_log("resume_publish_service success=true");
  }

  void on_timer()
  {
    if (trajectory_publish_paused_) {
      return;
    }
    if (!left_planner_.active() || !right_planner_.active() ||
      (neck_planning_enabled() && !neck_planner_.active()))
    {
      return;
    }
    if (require_joint_state_before_start_ && !joint_state_fresh()) {
      RCLCPP_WARN_THROTTLE(
        get_logger(), *get_clock(), 2000,
        "Skipping min-snap publish: /joint_states is stale or incomplete");
      return;
    }

    const auto now_time = now();
    const double elapsed = (now_time - trajectory_start_time_).seconds();
    auto left = left_planner_.sample(elapsed);
    auto right = right_planner_.sample(elapsed);
    auto neck_sample = neck_planner_.sample(elapsed);
    if (!clamp_sample_to_publish_limits(left, "left_publish") ||
      !clamp_sample_to_publish_limits(right, "right_publish") ||
      (neck_planning_enabled() && !clamp_neck_sample_to_publish_limits(neck_sample, "neck_publish")))
    {
      deactivate_trajectories("invalid_publish_sample");
      return;
    }
    if (robot_type_ == "sysmo32") {
      command_pub_->publish(
        make_sysmo32_command(left.position, right.position, speed_mode_, reserved_, neck_joint_));
      desired_joint_state_pub_->publish(make_sysmo32_desired_joint_state(
        now_time, left.position, right.position, left.velocity, right.velocity));
    } else {
      fill_upper_position_velocity(left, right, neck_sample, reusable_desired_position_, reusable_desired_velocity_);
      publish_reusable_fa_messages(now_time);
    }

    if (recorder_.enabled()) {
      UpperArray desired_position{};
      UpperArray desired_velocity{};
      UpperArray desired_acceleration{};
      UpperArray desired_jerk{};
      fill_upper_arrays(
        left, right, neck_sample, desired_position, desired_velocity, desired_acceleration, desired_jerk);
      recorder_.record(
        now_time.seconds(), latest_positions_, latest_velocities_, desired_position, desired_velocity,
        desired_acceleration, desired_jerk);
    }

    log_publish_sample(elapsed, left, right, neck_sample);
    maybe_deactivate_completed_trajectory(now_time, elapsed);
  }

  void maybe_deactivate_completed_trajectory(const rclcpp::Time & now_time, double elapsed)
  {
    if (!left_planner_.finished(elapsed) || !right_planner_.finished(elapsed) ||
      (neck_planning_enabled() && !neck_planner_.finished(elapsed)))
    {
      goal_reached_since_ = rclcpp::Time(0, 0, get_clock()->get_clock_type());
      return;
    }
    if (!has_goal_) {
      return;
    }

    if (!has_arm_joint_state_) {
      goal_reached_since_ = rclcpp::Time(0, 0, get_clock()->get_clock_type());
      return;
    }
    const ArmArray actual_left = latest_left_positions_;
    const ArmArray actual_right = latest_right_positions_;

    const double tolerance =
      std::isfinite(goal_reached_position_tolerance_rad_) && goal_reached_position_tolerance_rad_ > 0.0 ?
      goal_reached_position_tolerance_rad_ : 0.0;
    double left_error = 0.0;
    double right_error = 0.0;
    double neck_error = 0.0;
    for (std::size_t i = 0; i < active_arm_joint_count_; ++i) {
      left_error = std::max(left_error, std::abs(actual_left[i] - left_planner_.goal()[i]));
      right_error = std::max(right_error, std::abs(actual_right[i] - right_planner_.goal()[i]));
    }
    if (neck_planning_enabled() && has_neck_joint_state_) {
      neck_error = max_abs_neck_error(latest_neck_position_, neck_planner_.goal());
    }
    const double max_error = std::max({left_error, right_error, neck_error});
    if (max_error > tolerance) {
      goal_reached_since_ = rclcpp::Time(0, 0, get_clock()->get_clock_type());
      return;
    }

    if (goal_reached_since_.nanoseconds() == 0) {
      goal_reached_since_ = now_time;
      return;
    }
    const double hold_s =
      std::isfinite(goal_reached_hold_s_) && goal_reached_hold_s_ > 0.0 ? goal_reached_hold_s_ : 0.0;
    if ((now_time - goal_reached_since_).seconds() < hold_s) {
      return;
    }

    RCLCPP_INFO(
      get_logger(),
      "Min-snap target reached; deactivating trajectory. left_err=%.6f right_err=%.6f neck_err=%.6f tolerance=%.6f",
      left_error, right_error, neck_error, tolerance);
    write_run_log(
      "trajectory_completed target_seq=" + std::to_string(target_sequence_) +
      " elapsed_s=" + format_double(elapsed) +
      " left_max_abs_err=" + format_double(left_error) +
      " right_max_abs_err=" + format_double(right_error) +
      " neck_max_abs_err=" + format_double(neck_error) +
      " tolerance=" + format_double(tolerance));
    deactivate_trajectories("target_reached");
  }

  bool clamp_sample_to_publish_limits(ArmTrajectorySample & sample, const std::string & label)
  {
    const bool is_left = label.rfind("left_", 0) == 0;
    const ArmArray & lower = is_left ? left_lower_limits_ : right_lower_limits_;
    const ArmArray & upper = is_left ? left_upper_limits_ : right_upper_limits_;
    for (std::size_t i = 0; i < active_arm_joint_count_; ++i) {
      if (!std::isfinite(sample.position[i]) ||
        std::abs(sample.position[i]) > publish_position_abs_limit_rad_)
      {
        RCLCPP_ERROR(
          get_logger(),
          "Refusing min-snap %s: joint %zu position %.6f exceeds abs limit %.6f",
          label.c_str(), i, sample.position[i], publish_position_abs_limit_rad_);
        write_run_log(
          "safety_stop label=" + label +
          " reason=position_limit joint=" + std::to_string(i) +
          " value=" + format_double(sample.position[i]) +
          " limit=" + format_double(publish_position_abs_limit_rad_));
        return false;
      }
      const double clamped_position = std::clamp(sample.position[i], lower[i], upper[i]);
      if (clamped_position != sample.position[i]) {
        RCLCPP_WARN_THROTTLE(
          get_logger(), *get_clock(), 2000,
          "Clamped min-snap %s joint %zu from %.6f to %.6f within hard limits [%.6f, %.6f]",
          label.c_str(), i, sample.position[i], clamped_position, lower[i], upper[i]);
        sample.position[i] = clamped_position;
      }
      if (!std::isfinite(sample.velocity[i]) ||
        std::abs(sample.velocity[i]) > publish_velocity_abs_limit_rad_s_)
      {
        RCLCPP_ERROR(
          get_logger(),
          "Refusing min-snap %s: joint %zu velocity %.6f exceeds abs limit %.6f",
          label.c_str(), i, sample.velocity[i], publish_velocity_abs_limit_rad_s_);
        write_run_log(
          "safety_stop label=" + label +
          " reason=velocity_limit joint=" + std::to_string(i) +
          " value=" + format_double(sample.velocity[i]) +
          " limit=" + format_double(publish_velocity_abs_limit_rad_s_),
          "ERROR");
        return false;
      }
      if (!std::isfinite(sample.acceleration[i]) ||
        std::abs(sample.acceleration[i]) > publish_acceleration_abs_limit_rad_s2_)
      {
        RCLCPP_ERROR(
          get_logger(),
          "Refusing min-snap %s: joint %zu acceleration %.6f exceeds abs limit %.6f",
          label.c_str(), i, sample.acceleration[i], publish_acceleration_abs_limit_rad_s2_);
        write_run_log(
          "safety_stop label=" + label +
          " reason=acceleration_limit joint=" + std::to_string(i) +
          " value=" + format_double(sample.acceleration[i]) +
          " limit=" + format_double(publish_acceleration_abs_limit_rad_s2_),
          "ERROR");
        return false;
      }
    }
    return true;
  }

  bool clamp_neck_sample_to_publish_limits(NeckTrajectorySample & sample, const std::string & label)
  {
    for (std::size_t i = 0; i < kNeckJointCount; ++i) {
      if (!std::isfinite(sample.position[i]) ||
        std::abs(sample.position[i]) > publish_position_abs_limit_rad_)
      {
        RCLCPP_ERROR(
          get_logger(),
          "Refusing min-snap %s: neck joint %zu position %.6f exceeds abs limit %.6f",
          label.c_str(), i, sample.position[i], publish_position_abs_limit_rad_);
        write_run_log(
          "safety_stop label=" + label +
          " reason=position_limit joint=" + std::to_string(i) +
          " value=" + format_double(sample.position[i]) +
          " limit=" + format_double(publish_position_abs_limit_rad_),
          "ERROR");
        return false;
      }
      const double clamped_position = std::clamp(sample.position[i], neck_lower_limits_[i], neck_upper_limits_[i]);
      if (clamped_position != sample.position[i]) {
        RCLCPP_WARN_THROTTLE(
          get_logger(), *get_clock(), 2000,
          "Clamped min-snap %s neck joint %zu from %.6f to %.6f within hard limits [%.6f, %.6f]",
          label.c_str(), i, sample.position[i], clamped_position, neck_lower_limits_[i], neck_upper_limits_[i]);
        sample.position[i] = clamped_position;
      }
      if (!std::isfinite(sample.velocity[i]) ||
        std::abs(sample.velocity[i]) > publish_velocity_abs_limit_rad_s_)
      {
        RCLCPP_ERROR(
          get_logger(),
          "Refusing min-snap %s: neck joint %zu velocity %.6f exceeds abs limit %.6f",
          label.c_str(), i, sample.velocity[i], publish_velocity_abs_limit_rad_s_);
        write_run_log(
          "safety_stop label=" + label +
          " reason=velocity_limit joint=" + std::to_string(i) +
          " value=" + format_double(sample.velocity[i]) +
          " limit=" + format_double(publish_velocity_abs_limit_rad_s_),
          "ERROR");
        return false;
      }
      if (!std::isfinite(sample.acceleration[i]) ||
        std::abs(sample.acceleration[i]) > publish_acceleration_abs_limit_rad_s2_)
      {
        RCLCPP_ERROR(
          get_logger(),
          "Refusing min-snap %s: neck joint %zu acceleration %.6f exceeds abs limit %.6f",
          label.c_str(), i, sample.acceleration[i], publish_acceleration_abs_limit_rad_s2_);
        write_run_log(
          "safety_stop label=" + label +
          " reason=acceleration_limit joint=" + std::to_string(i) +
          " value=" + format_double(sample.acceleration[i]) +
          " limit=" + format_double(publish_acceleration_abs_limit_rad_s2_),
          "ERROR");
        return false;
      }
    }
    return true;
  }

  void clamp_goal_to_joint_limits(ArmArray & goal, const std::string & side)
  {
    const bool is_left = side == "left";
    const ArmArray & lower = is_left ? left_lower_limits_ : right_lower_limits_;
    const ArmArray & upper = is_left ? left_upper_limits_ : right_upper_limits_;
    for (std::size_t i = 0; i < active_arm_joint_count_; ++i) {
      double soft_lower = lower[i] + publish_joint_limit_margin_rad_;
      double soft_upper = upper[i] - publish_joint_limit_margin_rad_;
      if (soft_lower > soft_upper) {
        soft_lower = lower[i];
        soft_upper = upper[i];
      }
      const double original = goal[i];
      goal[i] = std::clamp(goal[i], soft_lower, soft_upper);
      if (goal[i] != original) {
        RCLCPP_WARN(
          get_logger(), "Clamped %s target joint %zu from %.6f to %.6f within [%.6f, %.6f]",
          side.c_str(), i, original, goal[i], soft_lower, soft_upper);
        write_run_log(
          "target_clamped side=" + side +
          " joint=" + std::to_string(i) +
          " original=" + format_double(original) +
          " clamped=" + format_double(goal[i]) +
          " lower=" + format_double(soft_lower) +
          " upper=" + format_double(soft_upper),
          "WARN");
      }
    }
  }

  void clamp_goal_to_neck_limits(NeckArray & goal)
  {
    for (std::size_t i = 0; i < kNeckJointCount; ++i) {
      double soft_lower = neck_lower_limits_[i] + publish_joint_limit_margin_rad_;
      double soft_upper = neck_upper_limits_[i] - publish_joint_limit_margin_rad_;
      if (soft_lower > soft_upper) {
        soft_lower = neck_lower_limits_[i];
        soft_upper = neck_upper_limits_[i];
      }
      const double original = goal[i];
      goal[i] = std::clamp(goal[i], soft_lower, soft_upper);
      if (goal[i] != original) {
        RCLCPP_WARN(
          get_logger(), "Clamped neck target joint %zu from %.6f to %.6f within [%.6f, %.6f]",
          i, original, goal[i], soft_lower, soft_upper);
        write_run_log(
          "target_clamped side=neck joint=" + std::to_string(i) +
          " original=" + format_double(original) +
          " clamped=" + format_double(goal[i]) +
          " lower=" + format_double(soft_lower) +
          " upper=" + format_double(soft_upper),
          "WARN");
      }
    }
  }

  ArmArray soft_lower_limits(const ArmArray & lower, const ArmArray & upper) const
  {
    ArmArray out{};
    for (std::size_t i = 0; i < kArmJointCount; ++i) {
      out[i] = lower[i] + publish_joint_limit_margin_rad_;
      if (out[i] > upper[i] - publish_joint_limit_margin_rad_) {
        out[i] = lower[i];
      }
    }
    return out;
  }

  ArmArray soft_upper_limits(const ArmArray & lower, const ArmArray & upper) const
  {
    ArmArray out{};
    for (std::size_t i = 0; i < kArmJointCount; ++i) {
      out[i] = upper[i] - publish_joint_limit_margin_rad_;
      if (lower[i] + publish_joint_limit_margin_rad_ > out[i]) {
        out[i] = upper[i];
      }
    }
    return out;
  }

  NeckArray soft_neck_lower_limits() const
  {
    NeckArray out{};
    for (std::size_t i = 0; i < kNeckJointCount; ++i) {
      out[i] = neck_lower_limits_[i] + publish_joint_limit_margin_rad_;
      if (out[i] > neck_upper_limits_[i] - publish_joint_limit_margin_rad_) {
        out[i] = neck_lower_limits_[i];
      }
    }
    return out;
  }

  NeckArray soft_neck_upper_limits() const
  {
    NeckArray out{};
    for (std::size_t i = 0; i < kNeckJointCount; ++i) {
      out[i] = neck_upper_limits_[i] - publish_joint_limit_margin_rad_;
      if (neck_lower_limits_[i] + publish_joint_limit_margin_rad_ > out[i]) {
        out[i] = neck_upper_limits_[i];
      }
    }
    return out;
  }

  void clamp_start_velocity_to_limit(ArmArray & velocity, double max_velocity, const std::string & side)
  {
    for (std::size_t i = 0; i < active_arm_joint_count_; ++i) {
      const double original = velocity[i];
      velocity[i] = std::clamp(velocity[i], -max_velocity, max_velocity);
      if (velocity[i] != original) {
        RCLCPP_WARN(
          get_logger(), "Clamped %s start velocity joint %zu from %.6f to %.6f for max_velocity %.6f",
          side.c_str(), i, original, velocity[i], max_velocity);
        write_run_log(
          "start_velocity_clamped side=" + side +
          " joint=" + std::to_string(i) +
          " original=" + format_double(original) +
          " clamped=" + format_double(velocity[i]) +
          " max_velocity=" + format_double(max_velocity),
          "WARN");
      }
    }
  }

  void clamp_neck_start_velocity_to_limit(NeckArray & velocity, double max_velocity)
  {
    for (std::size_t i = 0; i < kNeckJointCount; ++i) {
      const double original = velocity[i];
      velocity[i] = std::clamp(velocity[i], -max_velocity, max_velocity);
      if (velocity[i] != original) {
        RCLCPP_WARN(
          get_logger(), "Clamped neck start velocity joint %zu from %.6f to %.6f for max_velocity %.6f",
          i, original, velocity[i], max_velocity);
        write_run_log(
          "start_velocity_clamped side=neck joint=" + std::to_string(i) +
          " original=" + format_double(original) +
          " clamped=" + format_double(velocity[i]) +
          " max_velocity=" + format_double(max_velocity),
          "WARN");
      }
    }
  }

  void deactivate_trajectories(const std::string & reason)
  {
    left_planner_.deactivate();
    right_planner_.deactivate();
    neck_planner_.deactivate();
    has_goal_ = false;
    goal_reached_since_ = rclcpp::Time(0, 0, get_clock()->get_clock_type());
    if (timer_) {
      timer_->cancel();
    }
    write_run_log("trajectories_deactivated reason=" + reason);
  }

  void open_run_log()
  {
    if (!record_run_log_) {
      return;
    }
    try {
      std::filesystem::create_directories(run_log_output_dir_);
      const auto wall_now = std::chrono::system_clock::now();
      const auto time_t_now = std::chrono::system_clock::to_time_t(wall_now);
      std::tm tm_now{};
      localtime_r(&time_t_now, &tm_now);
      std::ostringstream path;
      path << run_log_output_dir_ << "/" << std::put_time(&tm_now, "%Y-%m-%d-%H--%M-%S") << ".log";
      run_log_path_ = path.str();
      run_log_file_.open(run_log_path_, std::ios::out | std::ios::trunc);
      if (!run_log_file_.is_open()) {
        RCLCPP_ERROR(get_logger(), "Failed to open min-snap run log: %s", run_log_path_.c_str());
        return;
      }
      RCLCPP_INFO(get_logger(), "Min-snap run log: %s", run_log_path_.c_str());
      write_run_log("log_opened path=" + run_log_path_);
    } catch (const std::exception & exc) {
      RCLCPP_ERROR(get_logger(), "Failed to create min-snap run log: %s", exc.what());
    }
  }

  static std::string wall_clock_log_time()
  {
    const auto wall_now = std::chrono::system_clock::now();
    const auto time_t_now = std::chrono::system_clock::to_time_t(wall_now);
    std::tm tm_now{};
    localtime_r(&time_t_now, &tm_now);
    const auto millis = std::chrono::duration_cast<std::chrono::milliseconds>(
      wall_now.time_since_epoch()).count() % 1000;

    std::ostringstream stream;
    stream << std::put_time(&tm_now, "%H:%M:%S") << "." << std::setw(3) << std::setfill('0') << millis;
    return stream.str();
  }

  void write_run_log(const std::string & message, const std::string & level = "INFO")
  {
    if (!run_log_file_.is_open()) {
      return;
    }
    run_log_file_ << "[" << level << "] " << wall_clock_log_time()
                  << " MainProcess min_snap.min_snap_node: " << message << "\n";
    run_log_file_.flush();
  }

  void write_run_log_throttled(const std::string & message)
  {
    if (!run_log_file_.is_open()) {
      return;
    }
    const auto now_time = now();
    if (last_run_log_time_.nanoseconds() != 0 &&
      (now_time - last_run_log_time_).seconds() < std::max(0.001, run_log_period_s_))
    {
      return;
    }
    last_run_log_time_ = now_time;
    write_run_log(message);
  }

  void write_run_log_warn_throttled(const std::string & key, const std::string & message)
  {
    if (!run_log_file_.is_open()) {
      return;
    }
    const auto now_time = now();
    const auto last_it = last_warn_log_times_.find(key);
    if (last_it != last_warn_log_times_.end() &&
      (now_time - last_it->second).seconds() < std::max(0.001, run_log_warn_throttle_s_))
    {
      return;
    }
    last_warn_log_times_[key] = now_time;
    write_run_log(message, "WARN");
  }

  void log_publish_sample(
    double elapsed,
    const ArmTrajectorySample & left,
    const ArmTrajectorySample & right,
    const NeckTrajectorySample & neck)
  {
    if (!run_log_file_.is_open()) {
      return;
    }
    const auto now_time = now();
    if (last_publish_log_time_.nanoseconds() != 0 &&
      (now_time - last_publish_log_time_).seconds() < std::max(0.001, run_log_period_s_))
    {
      return;
    }
    last_publish_log_time_ = now_time;
    ++publish_sequence_;
    const double left_error = max_abs_error(
      latest_positions_, left_arm_joint_names(), left.position, active_arm_joint_count_);
    const double right_error = max_abs_error(
      latest_positions_, right_arm_joint_names(), right.position, active_arm_joint_count_);
    const double neck_error = max_abs_neck_error(latest_neck_position_, neck.position);
    write_run_log(
      "publish seq=" + std::to_string(publish_sequence_) +
      " target_seq=" + std::to_string(target_sequence_) +
      " elapsed_s=" + format_double(elapsed) +
      " left_des_pos=" + format_arm(left.position, active_arm_joint_count_) +
      " right_des_pos=" + format_arm(right.position, active_arm_joint_count_) +
      " left_des_vel=" + format_arm(left.velocity, active_arm_joint_count_) +
      " right_des_vel=" + format_arm(right.velocity, active_arm_joint_count_) +
      " left_des_acc=" + format_arm(left.acceleration, active_arm_joint_count_) +
      " right_des_acc=" + format_arm(right.acceleration, active_arm_joint_count_) +
      " left_des_jerk=" + format_arm(left.jerk, active_arm_joint_count_) +
      " right_des_jerk=" + format_arm(right.jerk, active_arm_joint_count_) +
      " neck_des_pos=" + format_neck(neck.position) +
      " neck_des_vel=" + format_neck(neck.velocity) +
      " neck_des_acc=" + format_neck(neck.acceleration) +
      " neck_des_jerk=" + format_neck(neck.jerk) +
      " left_actual=" + format_arm_actual(latest_positions_, left_arm_joint_names(), active_arm_joint_count_) +
      " right_actual=" + format_arm_actual(latest_positions_, right_arm_joint_names(), active_arm_joint_count_) +
      " neck_actual=" + format_neck_actual(latest_positions_, neck_joint_names()) +
      " left_max_abs_err=" + format_double(left_error) +
      " right_max_abs_err=" + format_double(right_error) +
      " neck_max_abs_err=" + format_double(neck_error));
  }

  static void fill_upper_arrays(
    const ArmTrajectorySample & left,
    const ArmTrajectorySample & right,
    const NeckTrajectorySample & neck,
    UpperArray & position,
    UpperArray & velocity,
    UpperArray & acceleration,
    UpperArray & jerk)
  {
    for (std::size_t i = 0; i < kArmJointCount; ++i) {
      position[i] = left.position[i];
      velocity[i] = left.velocity[i];
      acceleration[i] = left.acceleration[i];
      jerk[i] = left.jerk[i];
      const std::size_t right_idx = i + kArmJointCount;
      position[right_idx] = right.position[i];
      velocity[right_idx] = right.velocity[i];
      acceleration[right_idx] = right.acceleration[i];
      jerk[right_idx] = right.jerk[i];
    }
    position[14] = neck.position[0];
    position[15] = neck.position[1];
    velocity[14] = neck.velocity[0];
    velocity[15] = neck.velocity[1];
    acceleration[14] = neck.acceleration[0];
    acceleration[15] = neck.acceleration[1];
    jerk[14] = neck.jerk[0];
    jerk[15] = neck.jerk[1];
  }

  static void fill_upper_position_velocity(
    const ArmTrajectorySample & left,
    const ArmTrajectorySample & right,
    const NeckTrajectorySample & neck,
    UpperArray & position,
    UpperArray & velocity)
  {
    for (std::size_t i = 0; i < kArmJointCount; ++i) {
      position[i] = left.position[i];
      velocity[i] = left.velocity[i];
      const std::size_t right_idx = i + kArmJointCount;
      position[right_idx] = right.position[i];
      velocity[right_idx] = right.velocity[i];
    }
    position[14] = neck.position[0];
    position[15] = neck.position[1];
    velocity[14] = neck.velocity[0];
    velocity[15] = neck.velocity[1];
  }

  void initialize_reusable_messages()
  {
    reusable_upper_command_msg_.data.resize(kUpperJointCount);
    const auto & names = upper_joint_names();
    reusable_desired_joint_state_msg_.name.assign(names.begin(), names.end());
    reusable_desired_joint_state_msg_.position.resize(kUpperJointCount);
    reusable_desired_joint_state_msg_.velocity.resize(kUpperJointCount);
    joint_state_name_slots_.reserve(kUpperJointCount);
    const auto & left_names = left_arm_joint_names();
    const auto & right_names = right_arm_joint_names();
    const auto & neck_names = neck_joint_names();
    for (std::size_t i = 0; i < kArmJointCount; ++i) {
      joint_state_name_slots_[left_names[i]] = i;
      joint_state_name_slots_[right_names[i]] = i + kArmJointCount;
    }
    for (std::size_t i = 0; i < kNeckJointCount; ++i) {
      joint_state_name_slots_[neck_names[i]] = i + 2 * kArmJointCount;
    }
    latest_neck_position_ = neck_default_;
  }

  void set_cached_joint_position(
    std::size_t slot,
    double position,
    std::array<bool, kArmJointCount> & left_seen,
    std::array<bool, kArmJointCount> & right_seen,
    std::array<bool, kNeckJointCount> & neck_seen)
  {
    if (slot < kArmJointCount) {
      latest_left_positions_[slot] = position;
      left_seen[slot] = true;
    } else if (slot < 2 * kArmJointCount) {
      const std::size_t index = slot - kArmJointCount;
      latest_right_positions_[index] = position;
      right_seen[index] = true;
    } else {
      const std::size_t index = slot - 2 * kArmJointCount;
      if (index < kNeckJointCount) {
        latest_neck_position_[index] = position;
        neck_seen[index] = true;
      }
    }
  }

  void set_cached_joint_velocity(std::size_t slot, double velocity)
  {
    if (slot < kArmJointCount) {
      latest_left_velocities_[slot] = velocity;
    } else if (slot < 2 * kArmJointCount) {
      latest_right_velocities_[slot - kArmJointCount] = velocity;
    } else {
      const std::size_t index = slot - 2 * kArmJointCount;
      if (index < kNeckJointCount) {
        latest_neck_velocities_[index] = velocity;
      }
    }
  }

  void publish_reusable_fa_messages(const rclcpp::Time & stamp)
  {
    std::copy(
      reusable_desired_position_.begin(), reusable_desired_position_.end(),
      reusable_upper_command_msg_.data.begin());
    command_pub_->publish(reusable_upper_command_msg_);

    reusable_desired_joint_state_msg_.header.stamp = stamp;
    std::copy(
      reusable_desired_position_.begin(), reusable_desired_position_.end(),
      reusable_desired_joint_state_msg_.position.begin());
    std::copy(
      reusable_desired_velocity_.begin(), reusable_desired_velocity_.end(),
      reusable_desired_joint_state_msg_.velocity.begin());
    desired_joint_state_pub_->publish(reusable_desired_joint_state_msg_);
  }

  double publish_hz_{1000.0};
  std::string robot_type_{"fa"};
  std::size_t active_arm_joint_count_{kArmJointCount};
  std::string target_topic_;
  std::string command_topic_;
  std::string desired_joint_states_topic_;
  std::string joint_states_topic_;
  std::string pause_publish_service_name_;
  std::string resume_publish_service_name_;
  PlannerConfig planner_config_;
  double default_expected_duration_s_{0.016};
  double default_max_velocity_rad_s_{1.5};
  double default_max_acceleration_rad_s2_{15.0};
  double replan_threshold_rad_{0.0005};
  double publish_position_abs_limit_rad_{4.0};
  double publish_joint_limit_margin_rad_{0.02};
  double publish_velocity_abs_limit_rad_s_{2.0};
  double publish_acceleration_abs_limit_rad_s2_{20.0};
  bool use_joint_state_velocity_{true};
  double joint_state_velocity_abs_limit_rad_s_{5.0};
  bool replan_from_joint_state_{false};
  double min_replan_interval_s_{0.016};
  bool require_joint_state_before_start_{true};
  double joint_state_timeout_s_{0.5};
  double goal_reached_position_tolerance_rad_{0.01};
  double goal_reached_hold_s_{0.05};
  bool record_tracking_{false};
  std::string tracking_output_dir_;
  bool record_run_log_{true};
  std::string run_log_output_dir_;
  double run_log_period_s_{0.1};
  double run_log_warn_throttle_s_{2.0};
  NeckArray neck_default_{0.0, 0.0};
  double speed_mode_{4.0};
  Sysmo32ReservedArray reserved_{0.0, 0.0, 0.0, 0.0};
  double neck_joint_{0.0};
  ArmArray left_lower_limits_{make_arm_array({-2.750, -0.288, -2.650, -2.050, -2.735, -0.469, -1.550})};
  ArmArray left_upper_limits_{make_arm_array({2.748, 3.342, 2.583, 1.440, 2.729, 0.482, 1.520})};
  ArmArray right_lower_limits_{make_arm_array({-2.750, -3.330, -2.570, -2.050, -2.730, -0.480, -1.536})};
  ArmArray right_upper_limits_{make_arm_array({2.750, 0.302, 2.660, 1.440, 2.730, 0.470, 1.532})};
  NeckArray neck_lower_limits_{make_neck_array({-1.570, -0.523})};
  NeckArray neck_upper_limits_{make_neck_array({1.570, 0.785})};

  rclcpp::Publisher<std_msgs::msg::Float64MultiArray>::SharedPtr command_pub_;
  rclcpp::Publisher<sensor_msgs::msg::JointState>::SharedPtr desired_joint_state_pub_;
  rclcpp::Subscription<min_snap::msg::MinSnapTarget>::SharedPtr target_sub_;
  rclcpp::Subscription<sensor_msgs::msg::JointState>::SharedPtr joint_state_sub_;
  rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr pause_publish_service_;
  rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr resume_publish_service_;
  rclcpp::TimerBase::SharedPtr timer_;

  MinSnapArmPlanner left_planner_;
  MinSnapArmPlanner right_planner_;
  MinSnapNeckPlanner neck_planner_;
  rclcpp::Time trajectory_start_time_{0, 0, RCL_ROS_TIME};
  rclcpp::Time last_joint_state_time_{0, 0, RCL_ROS_TIME};
  rclcpp::Time last_plan_time_{0, 0, RCL_ROS_TIME};
  rclcpp::Time goal_reached_since_{0, 0, RCL_ROS_TIME};
  rclcpp::Time last_run_log_time_{0, 0, RCL_ROS_TIME};
  rclcpp::Time last_publish_log_time_{0, 0, RCL_ROS_TIME};
  std::unordered_map<std::string, rclcpp::Time> last_warn_log_times_;
  bool has_arm_joint_state_{false};
  bool has_neck_joint_state_{false};
  bool has_goal_{false};
  bool trajectory_publish_paused_{false};
  ArmArray latest_left_goal_{};
  ArmArray latest_right_goal_{};
  NeckArray latest_neck_goal_{};

  std::unordered_map<std::string, double> latest_positions_;
  std::unordered_map<std::string, double> latest_velocities_;
  std::ofstream run_log_file_;
  std::string run_log_path_;
  std::uint64_t target_sequence_{0};
  std::uint64_t publish_sequence_{0};
  TrajectoryRecorder recorder_;
  UpperArray reusable_desired_position_{};
  UpperArray reusable_desired_velocity_{};
  std_msgs::msg::Float64MultiArray reusable_upper_command_msg_;
  sensor_msgs::msg::JointState reusable_desired_joint_state_msg_;
  std::unordered_map<std::string, std::size_t> joint_state_name_slots_;
  ArmArray latest_left_positions_{};
  ArmArray latest_right_positions_{};
  ArmArray latest_left_velocities_{};
  ArmArray latest_right_velocities_{};
  NeckArray latest_neck_position_{0.0, 0.0};
  NeckArray latest_neck_velocities_{0.0, 0.0};
};

}  // namespace min_snap

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<min_snap::MinSnapNode>());
  rclcpp::shutdown();
  return 0;
}
