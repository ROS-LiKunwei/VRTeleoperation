#include "fa_teleop/core.hpp"
#include "fa_teleop/planner.hpp"
#include <algorithm>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <thread>
#include <vector>

namespace {
double quantile(std::vector<double> v, double p) {
  if (v.empty())
    return 0;
  std::sort(v.begin(), v.end());
  return v[static_cast<size_t>(p * (v.size() - 1))];
}
} // namespace
int main(int argc, char **argv) {
  try {
    std::string bundle, output = "benchmark.json", profile = "precision_200hz";
    int steps = 10000, iterations = 0;
    double duration = 0;
    bool realtime = false;
    for (int i = 1; i < argc; ++i) {
      const std::string a = argv[i];
      if (a == "--bundle" && i + 1 < argc)
        bundle = argv[++i];
      else if (a == "--output" && i + 1 < argc)
        output = argv[++i];
      else if (a == "--steps" && i + 1 < argc)
        steps = std::stoi(argv[++i]);
      else if (a == "--profile" && i + 1 < argc)
        profile = argv[++i];
      else if (a == "--iterations" && i + 1 < argc)
        iterations = std::stoi(argv[++i]);
      else if (a == "--duration" && i + 1 < argc)
        duration = std::stod(argv[++i]);
      else if (a == "--realtime")
        realtime = true;
      else
        throw std::runtime_error("Unknown benchmark argument " + a);
    }
    if (steps <= 0 || bundle.empty())
      throw std::runtime_error("Positive steps and bundle required");
    fa::verify_bundle(bundle);
    auto c = fa::Config::load(bundle + "/config/" + profile + ".yaml");
    if (iterations)
      c.ik_iterations = iterations;
    if (duration)
      c.duration = duration;
    c.validate();
    fa::Kinematics kin(bundle + "/models/fa_description/urdf/fa_robot.urdf",
                       bundle + "/models/fa_robot.srdf", c);
    auto seed = fa::initial_pose(), actual = seed;
    fa::Plan plan;
    std::vector<double> costs, lateness, ik_times, plan_times;
    costs.reserve(steps);
    lateness.reserve(steps);
    size_t failures = 0, deadline_misses = 0, wakeup_misses = 0;
    std::array<double, 2> max_position{}, max_orientation{}, max_ik_position{},
        max_ik_orientation{};
    double squared_position = 0, max_velocity = 0, max_acceleration = 0;
    const double dt = 1 / c.control_hz;
    const auto clock_start = fa::Clock::now();
    auto wake = clock_start;
    for (int i = -50; i < steps; ++i) {
      const double t = std::max(0, i) * dt;
      fa::Upper desired = fa::initial_pose();
      // Known reachable continuous path, both arms, with a declared 10 s
      // sinusoidal period.
      for (int j = 0; j < 14; ++j)
        desired[j] +=
            0.02 * std::sin(2 * 3.141592653589793 * 0.1 * t + 0.15 * (j % 7)) -
            0.02 * std::sin(0.15 * (j % 7));
      const std::array<fa::Pose, 2> targets{
          kin.fk(desired.head<7>(), true),
          kin.fk(desired.segment<7>(7), false)};
      const double started = fa::steady_seconds();
      const auto solved = kin.solve(targets, seed);
      fa::Target target;
      target.mode = 2;
      target.q = solved.q;
      target.sequence = static_cast<uint64_t>(i + 51);
      target.duration = c.duration;
      target.velocity = c.velocity_limit;
      target.acceleration = c.acceleration_limit;
      const double activation = t + c.plan_lead;
      const auto from =
          plan.valid ? plan.sample(activation) : fa::stationary(seed);
      const double plan_started = fa::steady_seconds();
      fa::Plan next;
      bool planned =
          solved.valid && fa::build_plan(from, target, c, activation, next);
      const double plan_ms = (fa::steady_seconds() - plan_started) * 1000;
      if (planned) {
        seed = solved.q;
      } else if (i >= 0)
        ++failures;
      for (int k = 0; k < 5; ++k) {
        const double sample_time = t + k / 1000.0;
        if (planned && sample_time >= activation) {
          plan = next;
          planned = false;
        }
        const auto reference =
            plan.valid ? plan.sample(sample_time) : fa::stationary(actual);
        if (!fa::valid_reference(reference, c))
          throw std::runtime_error("Invalid sampled reference");
        actual += 0.25 * (reference.q - actual);
        max_velocity =
            std::max(max_velocity, reference.velocity.cwiseAbs().maxCoeff());
        max_acceleration = std::max(
            max_acceleration, reference.acceleration.cwiseAbs().maxCoeff());
      }
      if (planned)
        plan = next;
      const double cost = (fa::steady_seconds() - started) * 1000;
      if (i >= 0) {
        costs.push_back(cost);
        ik_times.push_back(solved.elapsed_ms);
        plan_times.push_back(plan_ms);
        if (cost > 5)
          ++deadline_misses;
        for (int arm = 0; arm < 2; ++arm) {
          const auto pose = kin.fk(actual.segment<7>(7 * arm), arm == 0);
          const double ep = (pose.p - targets[arm].p).norm(),
                       er = fa::rotation_error(pose.R, targets[arm].R);
          max_position[arm] = std::max(max_position[arm], ep);
          max_orientation[arm] = std::max(max_orientation[arm], er);
          max_ik_position[arm] =
              std::max(max_ik_position[arm], solved.position_error[arm]);
          max_ik_orientation[arm] =
              std::max(max_ik_orientation[arm], solved.orientation_error[arm]);
          squared_position += ep * ep;
        }
      }
      if (realtime && i >= 0) {
        wake += std::chrono::milliseconds(5);
        std::this_thread::sleep_until(wake);
        const double late = std::max(
            0.0,
            std::chrono::duration<double, std::milli>(fa::Clock::now() - wake)
                .count());
        lateness.push_back(late);
        if (late >= 5)
          ++wakeup_misses;
      }
      if (i == -1)
        wake = fa::Clock::now();
    }
    const double elapsed =
        std::chrono::duration<double>(fa::Clock::now() - clock_start).count();
    const bool accuracy =
        max_position[0] <= 0.001 && max_position[1] <= 0.001 &&
        max_orientation[0] <= 0.01 && max_orientation[1] <= 0.01;
    const bool passed = accuracy && failures == 0 && deadline_misses == 0 &&
                        (!realtime || wakeup_misses == 0);
    std::ofstream out(output);
    out << std::boolalpha << std::setprecision(12);
    out << "{\n\"release_id\":\"" << fa::release_id()
        << "\",\n\"steps\":" << steps << ",\n\"realtime\":" << realtime
        << ",\n\"elapsed_s\":" << elapsed
        << ",\n\"ik_iterations\":" << c.ik_iterations
        << ",\n\"trajectory_duration_s\":" << c.duration
        << ",\n\"path_joint_amplitude_rad\":0.02,\n\"path_frequency_hz\":0.1,"
           "\n\"plant\":\"kinematic first-order 0.25 per 1 ms, not physical "
           "hardware\",\n\"failed_steps\":"
        << failures << ",\n\"compute_deadline_misses\":" << deadline_misses
        << ",\n\"wakeup_deadline_misses\":" << wakeup_misses
        << ",\n\"compute_ms_p99\":" << quantile(costs, 0.99)
        << ",\n\"compute_ms_max\":" << quantile(costs, 1)
        << ",\n\"ik_ms_p99\":" << quantile(ik_times, .99)
        << ",\n\"planning_ms_p99\":" << quantile(plan_times, .99)
        << ",\n\"wakeup_ms_p99\":" << quantile(lateness, .99)
        << ",\n\"wakeup_ms_max\":" << quantile(lateness, 1)
        << ",\n\"position_max_m\":[" << max_position[0] << ','
        << max_position[1] << "],\n\"orientation_max_rad\":["
        << max_orientation[0] << ',' << max_orientation[1]
        << "],\n\"ik_position_max_m\":[" << max_ik_position[0] << ','
        << max_ik_position[1] << "],\n\"ik_orientation_max_rad\":["
        << max_ik_orientation[0] << ',' << max_ik_orientation[1]
        << "],\n\"position_rms_m\":"
        << std::sqrt(squared_position / (steps * 2))
        << ",\n\"velocity_max_rad_s\":" << max_velocity
        << ",\n\"acceleration_max_rad_s2\":" << max_acceleration
        << ",\n\"accuracy_pass\":" << accuracy
        << ",\n\"strict_all_deadlines_pass\":" << passed << "\n}\n";
    if (!out)
      throw std::runtime_error("Cannot write benchmark report");
    std::cout << "Benchmark: failures=" << failures
              << " position_max=" << std::max(max_position[0], max_position[1])
              << " compute_p99_ms=" << quantile(costs, .99)
              << " output=" << output << '\n';
    return failures == 0 && accuracy ? 0 : 2;
  } catch (const std::exception &e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
