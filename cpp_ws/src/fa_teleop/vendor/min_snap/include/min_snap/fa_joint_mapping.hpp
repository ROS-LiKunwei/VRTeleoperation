#pragma once

#include <array>
#include <string>
#include <unordered_map>
#include <vector>

#include "rclcpp/time.hpp"
#include "sensor_msgs/msg/joint_state.hpp"
#include "std_msgs/msg/float64_multi_array.hpp"

namespace min_snap
{

constexpr std::size_t kArmJointCount = 7;
constexpr std::size_t kUpperJointCount = 16;
constexpr std::size_t kSysmo32ArmJointCount = 6;
constexpr std::size_t kSysmo32CommandCount = 18;
constexpr std::size_t kSysmo32ReservedCount = 4;
constexpr std::size_t kNeckJointCount = 2;

using ArmArray = std::array<double, kArmJointCount>;
using UpperArray = std::array<double, kUpperJointCount>;
using Sysmo32CommandArray = std::array<double, kSysmo32CommandCount>;
using Sysmo32ReservedArray = std::array<double, kSysmo32ReservedCount>;
using NeckArray = std::array<double, kNeckJointCount>;

const std::array<std::string, kArmJointCount> & left_arm_joint_names();
const std::array<std::string, kArmJointCount> & right_arm_joint_names();
const std::array<std::string, kNeckJointCount> & neck_joint_names();
const std::array<std::string, kUpperJointCount> & upper_joint_names();

std_msgs::msg::Float64MultiArray make_upper_command(
  const ArmArray & left, const ArmArray & right, const NeckArray & neck);

std_msgs::msg::Float64MultiArray make_sysmo32_command(
  const ArmArray & left,
  const ArmArray & right,
  double speed_mode,
  const Sysmo32ReservedArray & reserved,
  double neck_joint);

sensor_msgs::msg::JointState make_desired_joint_state(
  const rclcpp::Time & stamp, const UpperArray & position, const UpperArray & velocity);

sensor_msgs::msg::JointState make_sysmo32_desired_joint_state(
  const rclcpp::Time & stamp, const ArmArray & left_position, const ArmArray & right_position,
  const ArmArray & left_velocity, const ArmArray & right_velocity);

bool extract_arm_positions(
  const std::unordered_map<std::string, double> & positions,
  ArmArray & left,
  ArmArray & right);

NeckArray extract_neck_or_default(
  const std::unordered_map<std::string, double> & positions,
  const NeckArray & default_neck);

}  // namespace min_snap
