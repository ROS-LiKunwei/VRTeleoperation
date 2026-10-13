#include "min_snap/trajectory_recorder.hpp"

#include <chrono>
#include <cmath>
#include <filesystem>
#include <iomanip>
#include <limits>
#include <sstream>

namespace min_snap
{

TrajectoryRecorder::~TrajectoryRecorder()
{
  close();
}

bool TrajectoryRecorder::open(const std::string & output_dir)
{
  close();
  std::filesystem::create_directories(output_dir);

  const auto now = std::chrono::system_clock::now();
  const auto time_t_now = std::chrono::system_clock::to_time_t(now);
  std::tm tm_now{};
  localtime_r(&time_t_now, &tm_now);

  std::ostringstream name;
  name << output_dir << "/min_snap_tracking_" << std::put_time(&tm_now, "%Y%m%d_%H%M%S") << ".csv";
  path_ = name.str();
  file_.open(path_, std::ios::out | std::ios::trunc);
  if (!file_.is_open()) {
    enabled_ = false;
    return false;
  }
  enabled_ = true;
  write_header();
  return true;
}

void TrajectoryRecorder::close()
{
  if (file_.is_open()) {
    file_.flush();
    file_.close();
  }
  enabled_ = false;
}

void TrajectoryRecorder::record(
  double time_s,
  const std::unordered_map<std::string, double> & actual_position,
  const std::unordered_map<std::string, double> & actual_velocity,
  const UpperArray & desired_position,
  const UpperArray & desired_velocity,
  const UpperArray & desired_acceleration,
  const UpperArray & desired_jerk)
{
  if (!enabled_ || !file_.is_open()) {
    return;
  }
  file_ << format(time_s);
  const auto & names = upper_joint_names();
  for (const auto & name : names) {
    const auto it = actual_position.find(name);
    file_ << "," << (it == actual_position.end() ? "" : format(it->second));
  }
  for (const auto & name : names) {
    const auto it = actual_velocity.find(name);
    file_ << "," << (it == actual_velocity.end() ? "" : format(it->second));
  }
  for (double value : desired_position) {
    file_ << "," << format(value);
  }
  for (double value : desired_velocity) {
    file_ << "," << format(value);
  }
  for (double value : desired_acceleration) {
    file_ << "," << format(value);
  }
  for (double value : desired_jerk) {
    file_ << "," << format(value);
  }
  for (std::size_t i = 0; i < names.size(); ++i) {
    const auto it = actual_position.find(names[i]);
    if (it == actual_position.end()) {
      file_ << ",";
    } else {
      file_ << "," << format(it->second - desired_position[i]);
    }
  }
  file_ << "\n";
}

std::string TrajectoryRecorder::format(double value)
{
  if (!std::isfinite(value)) {
    return "";
  }
  std::ostringstream stream;
  stream << std::setprecision(12) << value;
  return stream.str();
}

void TrajectoryRecorder::write_header()
{
  file_ << "time_s";
  const auto & names = upper_joint_names();
  for (const auto & name : names) {
    file_ << ",joint_states.position." << name;
  }
  for (const auto & name : names) {
    file_ << ",joint_states.velocity." << name;
  }
  for (const auto & name : names) {
    file_ << ",desired.position." << name;
  }
  for (const auto & name : names) {
    file_ << ",desired.velocity." << name;
  }
  for (const auto & name : names) {
    file_ << ",desired.acceleration." << name;
  }
  for (const auto & name : names) {
    file_ << ",desired.jerk." << name;
  }
  for (const auto & name : names) {
    file_ << ",error.position." << name;
  }
  file_ << "\n";
  file_.flush();
}

}  // namespace min_snap
