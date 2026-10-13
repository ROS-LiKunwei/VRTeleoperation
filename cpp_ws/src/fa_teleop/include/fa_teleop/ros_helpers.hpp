#pragma once
#include "fa_teleop/core.hpp"
#include <ament_index_cpp/get_package_share_directory.hpp>
#include <filesystem>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/joint_state.hpp>
#include <thread>
#include <yaml-cpp/yaml.h>
namespace fa {
inline std::string bundle_directory(rclcpp::Node &node) {
  const std::string dir = node.declare_parameter<std::string>(
      "bundle_dir",
      ament_index_cpp::get_package_share_directory("fa_teleop") + "/bundle");
  verify_bundle(dir);
  return std::filesystem::canonical(dir).string();
}
inline Config node_config(rclcpp::Node &node, const std::string &bundle) {
  const auto profile =
      node.declare_parameter<std::string>("profile", "precision_200hz");
  if (profile.find_first_not_of("abcdefghijklmnopqrstuvwxyz0123456789_") !=
      std::string::npos)
    throw std::runtime_error("Invalid locked profile name");
  const auto relative = "config/" + profile + ".yaml";
  const auto manifest = YAML::LoadFile(bundle + "/release.json");
  if (!manifest["files"][relative])
    throw std::runtime_error("Profile is not part of the compiled release");
  return Config::load(bundle + "/" + relative);
}
inline bool receive_feedback(const sensor_msgs::msg::JointState &msg,
                             Feedback &cached) {
  const double stamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9;
  return decode_feedback(msg.name, msg.position, msg.velocity, stamp,
                         steady_seconds(), cached);
}
inline bool feedback_fresh(const Feedback &fb, const Config &c) {
  return fb.valid && fresh(fb.received, fb.source_wall, steady_seconds(),
                           wall_seconds(), c.feedback_timeout, c);
}
class Periodic {
public:
  explicit Periodic(double hz)
      : period_(std::chrono::duration_cast<Clock::duration>(
            std::chrono::duration<double>(1 / hz))),
        next_(Clock::now()) {}
  // Absolute schedule. Do not execute a burst of stale catch-up iterations
  // after an overrun.
  double wait() {
    next_ += period_;
    std::this_thread::sleep_until(next_);
    const auto now = Clock::now();
    const double late = std::chrono::duration<double>(now - next_).count();
    if (now - next_ >= period_)
      next_ = now;
    return std::max(0.0, late);
  }

private:
  Clock::duration period_;
  Clock::time_point next_;
};
} // namespace fa
