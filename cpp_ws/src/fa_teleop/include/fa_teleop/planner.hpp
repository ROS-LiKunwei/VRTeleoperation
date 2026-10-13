#pragma once
#include "fa_teleop/core.hpp"
#include "min_snap/min_snap_planner.hpp"
namespace fa {
struct Target {
  Upper q{Upper::Zero()};
  uint64_t sequence{0};
  uint8_t mode{0};
  double received{0}, duration{0.04}, velocity{0.5}, acceleration{3};
};
struct Reference {
  Upper q{Upper::Zero()}, velocity{Upper::Zero()}, acceleration{Upper::Zero()};
  bool valid{false};
};
struct Plan {
  min_snap::MinSnapArmPlanner left{"left"}, right{"right"};
  min_snap::MinSnapNeckPlanner neck{"neck"};
  double start{0}, duration{0};
  uint64_t sequence{0};
  bool valid{false}, stopping{false};
  Reference sample(double now) const;
};
// Construct outside the sampler. On failure the old plan is never mutated.
bool build_plan(const Reference &from, const Target &target,
                const Config &config, double start, Plan &result);
Reference stationary(const Upper &q);
bool valid_reference(const Reference &r, const Config &c);
} // namespace fa
