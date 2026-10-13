#include "min_snap/fa_joint_mapping.hpp"

#include <algorithm>

#include "rclcpp/rclcpp.hpp"

namespace min_snap
{

const std::array<std::string, kArmJointCount> & left_arm_joint_names()
{
  static const std::array<std::string, kArmJointCount> names = {
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_yaw_joint",
    "left_wrist_pitch_joint",
    "left_wrist_roll_joint"};
  return names;
}

const std::array<std::string, kArmJointCount> & right_arm_joint_names()
{
  static const std::array<std::string, kArmJointCount> names = {
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_yaw_joint",
    "right_wrist_pitch_joint",
    "right_wrist_roll_joint"};
  return names;
}

const std::array<std::string, kNeckJointCount> & neck_joint_names()
{
  static const std::array<std::string, kNeckJointCount> names = {
    "neck_yaw_joint",
    "neck_pitch_joint"};
  return names;
}

const std::array<std::string, kUpperJointCount> & upper_joint_names()
{
  static const std::array<std::string, kUpperJointCount> names = {
    "left_shoulder_pitch_joint",
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

std_msgs::msg::Float64MultiArray make_upper_command(
  const ArmArray & left, const ArmArray & right, const NeckArray & neck)
{
  std_msgs::msg::Float64MultiArray msg;
  msg.data.resize(kUpperJointCount);
  std::copy(left.begin(), left.end(), msg.data.begin());
  std::copy(right.begin(), right.end(), msg.data.begin() + kArmJointCount);
  std::copy(neck.begin(), neck.end(), msg.data.begin() + 2 * kArmJointCount);
  return msg;
}

std_msgs::msg::Float64MultiArray make_sysmo32_command(
  const ArmArray & left,
  const ArmArray & right,
  double speed_mode,
  const Sysmo32ReservedArray & reserved,
  double neck_joint)
{
  std_msgs::msg::Float64MultiArray msg;
  msg.data.resize(kSysmo32CommandCount);
  std::copy_n(left.begin(), kSysmo32ArmJointCount, msg.data.begin());
  std::copy_n(right.begin(), kSysmo32ArmJointCount, msg.data.begin() + kSysmo32ArmJointCount);
  msg.data[12] = speed_mode;
  std::copy(reserved.begin(), reserved.end(), msg.data.begin() + 13);
  msg.data[17] = neck_joint;
  return msg;
}

sensor_msgs::msg::JointState make_desired_joint_state(
  const rclcpp::Time & stamp, const UpperArray & position, const UpperArray & velocity)
{
  sensor_msgs::msg::JointState msg;
  msg.header.stamp = stamp;
  const auto & names = upper_joint_names();
  msg.name.assign(names.begin(), names.end());
  msg.position.assign(position.begin(), position.end());
  msg.velocity.assign(velocity.begin(), velocity.end());
  return msg;
}

sensor_msgs::msg::JointState make_sysmo32_desired_joint_state(
  const rclcpp::Time & stamp, const ArmArray & left_position, const ArmArray & right_position,
  const ArmArray & left_velocity, const ArmArray & right_velocity)
{
  sensor_msgs::msg::JointState msg;
  msg.header.stamp = stamp;
  const auto & left_names = left_arm_joint_names();
  const auto & right_names = right_arm_joint_names();
  msg.name.reserve(kSysmo32ArmJointCount * 2);
  msg.position.reserve(kSysmo32ArmJointCount * 2);
  msg.velocity.reserve(kSysmo32ArmJointCount * 2);
  for (std::size_t i = 0; i < kSysmo32ArmJointCount; ++i) {
    msg.name.push_back(left_names[i]);
    msg.position.push_back(left_position[i]);
    msg.velocity.push_back(left_velocity[i]);
  }
  for (std::size_t i = 0; i < kSysmo32ArmJointCount; ++i) {
    msg.name.push_back(right_names[i]);
    msg.position.push_back(right_position[i]);
    msg.velocity.push_back(right_velocity[i]);
  }
  return msg;
}

bool extract_arm_positions(
  const std::unordered_map<std::string, double> & positions,
  ArmArray & left,
  ArmArray & right)
{
  const auto & left_names = left_arm_joint_names();
  const auto & right_names = right_arm_joint_names();
  for (std::size_t i = 0; i < kArmJointCount; ++i) {
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

NeckArray extract_neck_or_default(
  const std::unordered_map<std::string, double> & positions,
  const NeckArray & default_neck)
{
  NeckArray neck = default_neck;
  const auto & names = neck_joint_names();
  for (std::size_t i = 0; i < kNeckJointCount; ++i) {
    const auto it = positions.find(names[i]);
    if (it != positions.end()) {
      neck[i] = it->second;
    }
  }
  return neck;
}

}  // namespace min_snap
