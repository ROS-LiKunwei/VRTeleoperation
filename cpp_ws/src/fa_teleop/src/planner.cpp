#include "fa_teleop/planner.hpp"
#include <algorithm>
namespace fa {
namespace {
template <size_t N> std::array<double, N> array(const Upper &v, int offset) {
  std::array<double, N> a;
  for (size_t i = 0; i < N; ++i)
    a[i] = v[offset + i];
  return a;
}
template <size_t N>
void copy(const min_snap::JointTrajectorySample<N> &s, int offset,
          Reference &r) {
  for (size_t i = 0; i < N; ++i) {
    r.q[offset + i] = s.position[i];
    r.velocity[offset + i] = s.velocity[i];
    r.acceleration[offset + i] = s.acceleration[i];
  }
}
} // namespace
Reference stationary(const Upper &q) {
  Reference r;
  r.q = q;
  r.valid = q.allFinite();
  return r;
}
Reference Plan::sample(double now) const {
  Reference r;
  if (!valid)
    return r;
  const double t = std::max(0.0, now - start);
  copy(left.sample(t), 0, r);
  copy(right.sample(t), 7, r);
  copy(neck.sample(t), 14, r);
  r.valid = true;
  return r;
}
bool valid_reference(const Reference &r, const Config &c) {
  return r.valid && r.q.allFinite() && r.velocity.allFinite() &&
         r.acceleration.allFinite() &&
         (r.q.array() >= lower_limits().array() - 1e-9).all() &&
         (r.q.array() <= upper_limits().array() + 1e-9).all() &&
         r.velocity.cwiseAbs().maxCoeff() <= c.velocity_limit * 1.001 &&
         r.acceleration.cwiseAbs().maxCoeff() <= c.acceleration_limit * 1.001;
}
bool build_plan(const Reference &from, const Target &input, const Config &c,
                double start, Plan &result) {
  if (!from.valid || !from.q.allFinite() || !from.velocity.allFinite() ||
      !from.acceleration.allFinite() || !input.q.allFinite() ||
      !std::isfinite(input.duration) || input.duration <= 0 ||
      !std::isfinite(input.velocity) || input.velocity <= 0 ||
      !std::isfinite(input.acceleration) || input.acceleration <= 0 ||
      input.mode > 2)
    return false;
  const double velocity = std::min(input.velocity, c.velocity_limit),
               accel = std::min(input.acceleration, c.acceleration_limit);
  const auto lower = lower_limits(), upper = upper_limits();
  Upper goal = input.q;
  double duration = input.duration;
  if (input.mode == 0) {
    duration = std::max(0.04, 2 * from.velocity.cwiseAbs().maxCoeff() / accel);
    // Continuous braking target. A boundary-infeasible stop is rejected and
    // latched by the node.
    goal = from.q + 0.5 * duration * from.velocity;
  }
  if ((goal.array() < lower.array()).any() ||
      (goal.array() > upper.array()).any())
    return false;
  min_snap::PlannerConfig config;
  config.min_duration_s = 0.01;
  config.snap_weight_lambda = c.snap_weight;
  config.duration_scale_on_violation = 1.12;
  config.max_duration_search_iterations = c.duration_search_iterations;
  config.constraint_sample_count = c.constraint_samples;
  config.max_duration_on_violation_s = c.max_plan_duration;
  Plan p;
  p.start = start;
  p.sequence = input.sequence;
  p.stopping = input.mode == 0;
  auto plan_arm = [&](min_snap::MinSnapArmPlanner &planner, int offset) {
    return planner.plan(
        array<7>(from.q, offset), array<7>(from.velocity, offset),
        array<7>(from.acceleration, offset), array<7>(goal, offset), duration,
        velocity, accel, array<7>(lower, offset), array<7>(upper, offset),
        config);
  };
  if (!plan_arm(p.left, 0) || !plan_arm(p.right, 7) ||
      !p.neck.plan(array<2>(from.q, 14), array<2>(from.velocity, 14),
                   array<2>(from.acceleration, 14), array<2>(goal, 14),
                   duration, velocity, accel, array<2>(lower, 14),
                   array<2>(upper, 14), config))
    return false;
  p.duration =
      std::max({p.left.duration(), p.right.duration(), p.neck.duration()});
  p.valid = true;
  // Position constraints are also checked at output frequency, independent of
  // upstream samples.
  const int n = std::max(c.constraint_samples,
                         static_cast<int>(std::ceil(p.duration * c.sample_hz)));
  for (int i = 0; i <= n; ++i)
    if (!valid_reference(p.sample(start + p.duration * i / n), c))
      return false;
  result = p;
  return true;
}
} // namespace fa
