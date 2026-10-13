#pragma once

#include <array>
#include <cstddef>

namespace min_snap
{

struct TrajectoryState
{
  double position{0.0};
  double velocity{0.0};
  double acceleration{0.0};
  double jerk{0.0};
};

class MinSnapTrajectory
{
public:
  bool solve(
    double q0,
    double v0,
    double a0,
    double q1,
    double v1,
    double a1,
    double duration,
    double snap_weight_lambda = 1.0);

  TrajectoryState sample(double t) const;

  double duration() const { return duration_s_; }
  const std::array<double, 8> & coefficients() const { return coefficients_; }

  double max_abs_velocity(std::size_t sample_count) const;
  double max_abs_acceleration(std::size_t sample_count) const;

private:
  std::array<double, 8> coefficients_{};
  double duration_s_{0.0};
};

}  // namespace min_snap
