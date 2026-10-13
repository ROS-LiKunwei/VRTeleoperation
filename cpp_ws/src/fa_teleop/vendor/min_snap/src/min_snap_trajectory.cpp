#include "min_snap/min_snap_trajectory.hpp"

#include <algorithm>
#include <array>
#include <cmath>

namespace min_snap
{

namespace
{

template <std::size_t N>
bool solve_linear_system(std::array<std::array<double, N + 1>, N> & a, std::array<double, N> & x)
{
  for (std::size_t col = 0; col < N; ++col) {
    std::size_t pivot = col;
    double best = std::abs(a[col][col]);
    for (std::size_t row = col + 1; row < N; ++row) {
      const double value = std::abs(a[row][col]);
      if (value > best) {
        best = value;
        pivot = row;
      }
    }
    if (best < 1e-12) {
      return false;
    }
    if (pivot != col) {
      for (std::size_t k = col; k < N + 1; ++k) {
        std::swap(a[col][k], a[pivot][k]);
      }
    }
    const double diag = a[col][col];
    for (std::size_t k = col; k < N + 1; ++k) {
      a[col][k] /= diag;
    }
    for (std::size_t row = 0; row < N; ++row) {
      if (row == col) {
        continue;
      }
      const double factor = a[row][col];
      for (std::size_t k = col; k < N + 1; ++k) {
        a[row][k] -= factor * a[col][k];
      }
    }
  }
  for (std::size_t row = 0; row < N; ++row) {
    x[row] = a[row][N];
  }
  return true;
}

double monomial_integral(std::size_t power)
{
  return 1.0 / static_cast<double>(power + 1);
}

}  // namespace

bool MinSnapTrajectory::solve(
  double q0,
  double v0,
  double a0,
  double q1,
  double v1,
  double a1,
  double duration,
  double snap_weight_lambda)
{
  if (!std::isfinite(duration) || duration <= 1e-6) {
    return false;
  }
  duration_s_ = duration;
  coefficients_.fill(0.0);

  snap_weight_lambda = std::clamp(snap_weight_lambda, 0.0, 1.0);
  const double line_weight = 1.0 - snap_weight_lambda;

  std::array<std::array<double, 8>, 8> h{};
  std::array<double, 8> g{};

  for (std::size_t i = 0; i < 8; ++i) {
    h[i][i] += 1e-9;
  }
  for (std::size_t i = 4; i < 8; ++i) {
    const double di = static_cast<double>(i * (i - 1) * (i - 2) * (i - 3));
    for (std::size_t j = 4; j < 8; ++j) {
      const double dj = static_cast<double>(j * (j - 1) * (j - 2) * (j - 3));
      h[i][j] += 2.0 * snap_weight_lambda * di * dj * monomial_integral(i + j - 8);
    }
  }
  const double line_delta = q1 - q0;
  for (std::size_t i = 0; i < 8; ++i) {
    for (std::size_t j = 0; j < 8; ++j) {
      h[i][j] += 2.0 * line_weight * monomial_integral(i + j);
    }
    g[i] += -2.0 * line_weight *
      (q0 * monomial_integral(i) + line_delta * monomial_integral(i + 1));
  }

  std::array<std::array<double, 8>, 6> constraints{};
  std::array<double, 6> rhs{};
  constraints[0][0] = 1.0;
  rhs[0] = q0;
  constraints[1][1] = 1.0;
  rhs[1] = v0 * duration_s_;
  constraints[2][2] = 2.0;
  rhs[2] = a0 * duration_s_ * duration_s_;
  for (std::size_t i = 0; i < 8; ++i) {
    constraints[3][i] = 1.0;
  }
  rhs[3] = q1;
  for (std::size_t i = 1; i < 8; ++i) {
    constraints[4][i] = static_cast<double>(i);
  }
  rhs[4] = v1 * duration_s_;
  for (std::size_t i = 2; i < 8; ++i) {
    constraints[5][i] = static_cast<double>(i * (i - 1));
  }
  rhs[5] = a1 * duration_s_ * duration_s_;

  constexpr std::size_t kSystemSize = 14;
  std::array<std::array<double, kSystemSize + 1>, kSystemSize> system{};
  for (std::size_t row = 0; row < 8; ++row) {
    for (std::size_t col = 0; col < 8; ++col) {
      system[row][col] = h[row][col];
    }
    for (std::size_t c = 0; c < 6; ++c) {
      system[row][8 + c] = constraints[c][row];
    }
    system[row][kSystemSize] = -g[row];
  }
  for (std::size_t row = 0; row < 6; ++row) {
    for (std::size_t col = 0; col < 8; ++col) {
      system[8 + row][col] = constraints[row][col];
    }
    system[8 + row][kSystemSize] = rhs[row];
  }

  std::array<double, kSystemSize> solution{};
  if (!solve_linear_system(system, solution)) {
    return false;
  }
  for (std::size_t i = 0; i < 8; ++i) {
    coefficients_[i] = solution[i];
  }
  return true;
}

TrajectoryState MinSnapTrajectory::sample(double t) const
{
  const double x = std::clamp(t, 0.0, duration_s_);
  const double tau = duration_s_ > 0.0 ? x / duration_s_ : 0.0;
  std::array<double, 8> powers{};
  powers[0] = 1.0;
  for (std::size_t i = 1; i < powers.size(); ++i) {
    powers[i] = powers[i - 1] * tau;
  }

  TrajectoryState state;
  for (std::size_t i = 0; i < 8; ++i) {
    state.position += coefficients_[i] * powers[i];
  }
  for (std::size_t i = 1; i < 8; ++i) {
    state.velocity += static_cast<double>(i) * coefficients_[i] * powers[i - 1];
  }
  state.velocity /= duration_s_;
  for (std::size_t i = 2; i < 8; ++i) {
    state.acceleration +=
      static_cast<double>(i * (i - 1)) * coefficients_[i] * powers[i - 2];
  }
  state.acceleration /= duration_s_ * duration_s_;
  for (std::size_t i = 3; i < 8; ++i) {
    state.jerk +=
      static_cast<double>(i * (i - 1) * (i - 2)) * coefficients_[i] * powers[i - 3];
  }
  state.jerk /= duration_s_ * duration_s_ * duration_s_;
  return state;
}

double MinSnapTrajectory::max_abs_velocity(std::size_t sample_count) const
{
  sample_count = std::max<std::size_t>(2, sample_count);
  double peak = 0.0;
  for (std::size_t i = 0; i < sample_count; ++i) {
    const double t = duration_s_ * static_cast<double>(i) / static_cast<double>(sample_count - 1);
    peak = std::max(peak, std::abs(sample(t).velocity));
  }
  return peak;
}

double MinSnapTrajectory::max_abs_acceleration(std::size_t sample_count) const
{
  sample_count = std::max<std::size_t>(2, sample_count);
  double peak = 0.0;
  for (std::size_t i = 0; i < sample_count; ++i) {
    const double t = duration_s_ * static_cast<double>(i) / static_cast<double>(sample_count - 1);
    peak = std::max(peak, std::abs(sample(t).acceleration));
  }
  return peak;
}

}  // namespace min_snap
