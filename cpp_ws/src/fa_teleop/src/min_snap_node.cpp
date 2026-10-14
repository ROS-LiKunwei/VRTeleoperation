#include "fa_teleop/msg/joint_target.hpp"
#include "fa_teleop/planner.hpp"
#include "fa_teleop/ros_helpers.hpp"
#include <iostream>
#include <std_msgs/msg/float64_multi_array.hpp>
#include <std_msgs/msg/string.hpp>
#include <std_srvs/srv/trigger.hpp>

namespace fa {
class MinSnapNode final : public rclcpp::Node {
public:
  MinSnapNode() : Node("fa_min_snap") {
    const auto bundle = bundle_directory(*this);
    config_ = node_config(*this, bundle);
    enable_ = declare_parameter<bool>("enable_motion", false);
    if (config_.control_backend == "mujoco") {
      backend_sub_ = create_subscription<std_msgs::msg::String>(
          "fa_cpp/backend_status", rclcpp::QoS(1).reliable(),
          [this](std_msgs::msg::String::ConstSharedPtr m) {
            if (m->data.rfind(deployment_id(config_) + "|READY", 0) != 0) {
              trip(9);
              return;
            }
            backend_seen_ = steady_seconds();
          });
    }
    output_ = create_publisher<std_msgs::msg::Float64MultiArray>(
        "upper_position_controller/commands", rclcpp::QoS(1).reliable());
    desired_ = create_publisher<sensor_msgs::msg::JointState>(
        "fa_cpp/desired_joint_states", rclcpp::SensorDataQoS().keep_last(1));
    status_ = create_publisher<std_msgs::msg::String>(
        "fa_cpp/executor_status", rclcpp::QoS(1).reliable());
    feedback_sub_ = create_subscription<sensor_msgs::msg::JointState>(
        "joint_states", rclcpp::SensorDataQoS().keep_last(1),
        [this](sensor_msgs::msg::JointState::ConstSharedPtr m) {
          if (receive_feedback(*m, fb_callback_))
            feedback_.write(fb_callback_);
        });
    target_sub_ = create_subscription<fa_teleop::msg::JointTarget>(
        "fa_cpp/target", rclcpp::QoS(1).reliable(),
        [this](fa_teleop::msg::JointTarget::ConstSharedPtr m) {
          const double stamp =
              m->header.stamp.sec + m->header.stamp.nanosec * 1e-9;
          if (m->release_id != deployment_id(config_) ||
              m->header.frame_id != "pelvis" || m->mode > 2) {
            trip(1);
            return;
          }
          if (m->sequence <= last_sequence_ ||
              wall_seconds() - stamp > config_.target_timeout ||
              stamp - wall_seconds() > config_.timestamp_skew)
            return;
          Target t;
          t.sequence = m->sequence;
          t.mode = m->mode;
          t.received = steady_seconds();
          t.duration = m->duration_s;
          t.velocity = m->velocity_limit;
          t.acceleration = m->acceleration_limit;
          for (int i = 0; i < 7; ++i) {
            t.q[i] = m->left[i];
            t.q[i + 7] = m->right[i];
          }
          t.q[14] = m->neck[0];
          t.q[15] = m->neck[1];
          if (!t.q.allFinite() ||
              (t.q.array() < lower_limits().array()).any() ||
              (t.q.array() > upper_limits().array()).any() ||
              !std::isfinite(t.duration) || t.duration <= 0 ||
              t.duration > config_.max_plan_duration + 10 ||
              !std::isfinite(t.velocity) || t.velocity <= 0 ||
              t.velocity > config_.velocity_limit ||
              !std::isfinite(t.acceleration) || t.acceleration <= 0 ||
              t.acceleration > config_.acceleration_limit) {
            trip(2);
            return;
          }
          last_sequence_ = m->sequence;
          targets_.write(t);
        });
    reset_ = create_service<std_srvs::srv::Trigger>(
        "fa_cpp/executor_reset",
        [this](std_srvs::srv::Trigger::Request::SharedPtr,
               std_srvs::srv::Trigger::Response::SharedPtr r) {
          reset_requested_ = true;
          r->success = true;
          r->message =
              "Executor reset queued; explicit initialization required";
        });
    report_ = create_wall_timer(std::chrono::milliseconds(100), [this] {
      const auto publishers =
          get_publishers_info_by_topic(output_->get_topic_name());
      ownership_ok_ = publishers.size() == 1;
      if (publishers.size() > 1)
        trip(6);
      std_msgs::msg::String m;
      m.data = deployment_id(config_) + (fault_ ? "|FAULT" : "|READY") + "|" +
               std::to_string(plans_.load()) +
               "|late=" + std::to_string(late_plans_.load()) +
               "|samples=" + std::to_string(samples_.load()) +
               "|reason=" + std::to_string(fault_reason_.load());
      status_->publish(m);
    });
    running_ = true;
    planning_ = std::thread(&MinSnapNode::plan_loop, this);
    sampling_ = std::thread(&MinSnapNode::sample_loop, this);
  }
  ~MinSnapNode() override {
    running_ = false;
    if (planning_.joinable())
      planning_.join();
    if (sampling_.joinable())
      sampling_.join();
  }

private:
  bool backend_ready() const {
    return config_.control_backend == "real" ||
           steady_seconds() - backend_seen_.load() < 0.5;
  }
  void trip(int reason) {
    fault_reason_ = reason;
    fault_ = true;
  }
  void plan_loop() {
    Periodic schedule(config_.control_hz);
    Target target, last;
    Feedback fb;
    Plan current;
    uint64_t handled = 0, plan_generation = 0;
    bool stopping = false, initialized = false;
    while (running_ && rclcpp::ok()) {
      if (reset_requested_.exchange(false)) {
        current = Plan{};
        plan_.write(current);
        initialized = false;
        stopping = false;
        handled = 0;
        fault_ = false;
        fault_reason_ = 0;
        last_sequence_ = 0;
      }
      feedback_.read(fb);
      targets_.read(target);
      const double now_s = steady_seconds();
      if (!fault_ && enable_ && feedback_fresh(fb, config_) &&
          backend_ready()) {
        const bool stale = target.sequence == 0 ||
                           now_s - target.received > config_.target_timeout;
        if (initialized && stale && !stopping) {
          target = last;
          target.mode = 0;
          target.sequence = handled + 1;
          target.received = now_s;
        } else if (stale) {
          schedule.wait();
          continue;
        }
        if (!initialized && target.mode != 1) {
          schedule.wait();
          continue;
        }
        const bool new_target =
            target.sequence != handled || target.mode != last.mode;
        const bool same_goal =
            initialized && target.mode == last.mode &&
            (target.q - last.q).cwiseAbs().maxCoeff() < 1e-10;
        if (new_target && !same_goal && !(stopping && target.mode == 0)) {
          // Long initialization is planned once before output starts; tracking
          // retains the normal 4 ms submission budget.
          const double activation =
              now_s + (target.mode == 1 ? config_.initialization_plan_lead
                                        : config_.plan_lead);
          Reference from =
              current.valid ? current.sample(activation) : stationary(fb.q);
          Plan next;
          // Initialization uses its own duration ceiling without weakening
          // tracking limits.
          Config plan_config = config_;
          if (target.mode == 1)
            plan_config.max_plan_duration = std::max(
                config_.max_plan_duration, config_.initialization_duration + 5);
          if (build_plan(from, target, plan_config, activation, next)) {
            if (steady_seconds() < activation) {
              next.sequence = ++plan_generation;
              plan_.write(next);
              current = next;
              last = target;
              handled = target.sequence;
              initialized = true;
              stopping = target.mode == 0;
              consecutive_late_ = 0;
              ++plans_;
            } else {
              ++late_plans_;
              if (++consecutive_late_ >= 3)
                trip(4);
            }
          } else
            trip(3);
        } else if (new_target) {
          last = target;
          handled = target.sequence;
        }
      }
      schedule.wait();
    }
  }
  void sample_loop() {
    Periodic schedule(config_.sample_hz);
    Plan incoming, active, pending;
    Feedback fb;
    std_msgs::msg::Float64MultiArray command;
    command.data.resize(16);
    sensor_msgs::msg::JointState desired;
    desired.name.assign(joint_names().begin(), joint_names().end());
    desired.position.resize(16);
    desired.velocity.resize(16);
    while (running_ && rclcpp::ok()) {
      const double now_s = steady_seconds();
      if (config_.control_backend == "mujoco" && samples_ > 0 &&
          !backend_ready())
        trip(9);
      feedback_.read(fb);
      if (plan_.read(incoming)) {
        if (!incoming.valid) {
          active = Plan{};
          pending = Plan{};
        } else if (incoming.sequence != active.sequence &&
                   incoming.sequence != pending.sequence)
          pending = incoming;
      }
      if (pending.valid && now_s >= pending.start) {
        active = pending;
        pending = Plan{};
      }
      if (enable_ && !fault_ && active.valid && samples_ > 0 &&
          !feedback_fresh(fb, config_))
        trip(8);
      if (enable_ && ownership_ok_ && !fault_ && backend_ready() &&
          feedback_fresh(fb, config_) && active.valid) {
        const auto reference = active.sample(now_s);
        if (!valid_reference(reference, config_)) {
          trip(5);
          schedule.wait();
          continue;
        }
        if ((reference.q - fb.q).head<14>().cwiseAbs().maxCoeff() >
            config_.max_tracking_error) {
          trip(7);
          schedule.wait();
          continue;
        }
        for (int i = 0; i < 16; ++i) {
          command.data[i] = reference.q[i];
          desired.position[i] = reference.q[i];
          desired.velocity[i] = reference.velocity[i];
        }
        output_->publish(command);
        desired.header.stamp = now();
        desired_->publish(desired);
        ++samples_;
      }
      schedule.wait();
    }
  }
  Config config_;
  bool enable_{false};
  Snapshot<Target> targets_;
  Snapshot<Feedback> feedback_;
  Snapshot<Plan> plan_;
  Feedback fb_callback_;
  std::atomic<uint64_t> last_sequence_{0}, plans_{0}, late_plans_{0},
      samples_{0};
  int consecutive_late_{0};
  std::atomic<bool> running_{false}, fault_{false}, reset_requested_{false};
  std::atomic<bool> ownership_ok_{false};
  std::atomic<int> fault_reason_{0};
  std::atomic<double> backend_seen_{0};
  std::thread planning_, sampling_;
  rclcpp::Publisher<std_msgs::msg::Float64MultiArray>::SharedPtr output_;
  rclcpp::Publisher<sensor_msgs::msg::JointState>::SharedPtr desired_;
  rclcpp::Publisher<std_msgs::msg::String>::SharedPtr status_;
  rclcpp::Subscription<sensor_msgs::msg::JointState>::SharedPtr feedback_sub_;
  rclcpp::Subscription<fa_teleop::msg::JointTarget>::SharedPtr target_sub_;
  rclcpp::Subscription<std_msgs::msg::String>::SharedPtr backend_sub_;
  rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr reset_;
  rclcpp::TimerBase::SharedPtr report_;
};
} // namespace fa
int main(int argc, char **argv) {
  rclcpp::init(argc, argv);
  try {
    auto node = std::make_shared<fa::MinSnapNode>();
    rclcpp::spin(node);
  } catch (const std::exception &e) {
    std::cerr << e.what() << '\n';
    rclcpp::shutdown();
    return 1;
  }
  rclcpp::shutdown();
  return 0;
}
