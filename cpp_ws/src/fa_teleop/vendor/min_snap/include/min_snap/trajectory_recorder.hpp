#pragma once

#include <array>
#include <fstream>
#include <string>
#include <unordered_map>

#include "min_snap/fa_joint_mapping.hpp"

namespace min_snap
{

class TrajectoryRecorder
{
public:
  TrajectoryRecorder() = default;
  ~TrajectoryRecorder();

  bool open(const std::string & output_dir);
  void close();
  bool enabled() const { return enabled_; }
  const std::string & path() const { return path_; }

  void record(
    double time_s,
    const std::unordered_map<std::string, double> & actual_position,
    const std::unordered_map<std::string, double> & actual_velocity,
    const UpperArray & desired_position,
    const UpperArray & desired_velocity,
    const UpperArray & desired_acceleration,
    const UpperArray & desired_jerk);

private:
  static std::string format(double value);
  void write_header();

  bool enabled_{false};
  std::ofstream file_;
  std::string path_;
};

}  // namespace min_snap
