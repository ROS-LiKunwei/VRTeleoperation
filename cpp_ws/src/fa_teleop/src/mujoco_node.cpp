#include "fa_teleop/mujoco_engine.hpp"
#include "fa_teleop/ros_helpers.hpp"
#include <iostream>
#include <std_msgs/msg/float64_multi_array.hpp>
#include <std_msgs/msg/int32.hpp>
#include <std_msgs/msg/string.hpp>

namespace fa {
struct MujocoCommand {
  Upper q{Upper::Zero()};
  uint64_t sequence{0};
};
class MujocoNode final : public rclcpp::Node {
public:
  MujocoNode() : Node("fa_mujoco") {
    const auto bundle = bundle_directory(*this);
    config_ = node_config(*this, bundle);
    if (config_.control_backend != "mujoco")
      throw std::runtime_error(
          "fa_mujoco_node requires control_backend:=mujoco");
    engine_ = std::make_unique<MujocoEngine>(bundle);
    pub_ = create_publisher<sensor_msgs::msg::JointState>(
        "joint_states", rclcpp::SensorDataQoS().keep_last(1));
    status_ = create_publisher<std_msgs::msg::String>(
        "fa_cpp/backend_status", rclcpp::QoS(1).reliable());
    sub_ = create_subscription<std_msgs::msg::Float64MultiArray>(
        "upper_position_controller/commands", rclcpp::QoS(1).reliable(),
        [this](std_msgs::msg::Float64MultiArray::ConstSharedPtr m) {
          if (m->data.size() != 16) {
            ++rejected_;
            return;
          }
          MujocoCommand c;
          for (size_t i = 0; i < 16; ++i)
            c.q[i] = m->data[i];
          if (!c.q.allFinite() ||
              (c.q.array() < lower_limits().array()).any() ||
              (c.q.array() > upper_limits().array()).any()) {
            ++rejected_;
            return;
          }
          c.sequence = ++received_;
          commands_.write(c);
        });
    for (int i = 0; i < 2; ++i)
      hands_[i] = create_subscription<std_msgs::msg::Int32>(
          i == 0 ? "left_topic_to_hand" : "right_topic_to_hand", 10,
          [this, i](std_msgs::msg::Int32::ConstSharedPtr m) {
            if (m->data == 20 || m->data == 21 || m->data == 22)
              hand_commands_[i] = m->data;
          });
    report_ = create_wall_timer(std::chrono::milliseconds(100), [this] {
      if (get_publishers_info_by_topic(pub_->get_topic_name()).size() > 1)
        conflict_ = true;
      std_msgs::msg::String m;
      m.data = deployment_id(config_) + (conflict_ ? "|FAULT" : "|READY") +
               "|engine=" + mj_versionString() +
               "|feedback=" + std::to_string(samples_.load()) +
               "|applied=" + std::to_string(applied_.load()) +
               "|rejected=" + std::to_string(rejected_.load());
      status_->publish(m);
    });
    running_ = true;
    worker_ = std::thread([this] {
      try {
        run();
      } catch (const std::exception &e) {
        conflict_ = true;
        RCLCPP_ERROR(get_logger(), "MuJoCo worker: %s", e.what());
      }
    });
    RCLCPP_INFO(get_logger(),
                "MuJoCo %s, kinematic mirror, namespace=%s, 16-joint feedback "
                "at %.0f Hz",
                mj_versionString(), get_namespace(), config_.sample_hz);
  }
  ~MujocoNode() override {
    running_ = false;
    if (worker_.joinable())
      worker_.join();
  }

private:
  void run() {
    Periodic clock(config_.sample_hz);
    MujocoCommand command;
    uint64_t last = 0;
    double previous = steady_seconds();
    sensor_msgs::msg::JointState state;
    state.name.assign(joint_names().begin(), joint_names().end());
    state.position.resize(16);
    state.velocity.resize(16);
    while (running_ && rclcpp::ok()) {
      if (conflict_) {
        clock.wait();
        continue;
      }
      commands_.read(command);
      if (command.sequence != last) {
        if (engine_->apply(command.q)) {
          last = command.sequence;
          ++applied_;
        } else
          ++rejected_;
      }
      engine_->hand(true, hand_commands_[0]);
      engine_->hand(false, hand_commands_[1]);
      const double t = steady_seconds();
      engine_->forward(t - previous);
      previous = t;
      const auto q = engine_->positions(), v = engine_->velocities();
      state.header.stamp = now();
      state.header.frame_id = "pelvis";
      for (int i = 0; i < 16; ++i) {
        state.position[i] = q[i];
        state.velocity[i] = v[i];
      }
      pub_->publish(state);
      ++samples_;
      clock.wait();
    }
  }
  Config config_;
  std::unique_ptr<MujocoEngine> engine_;
  Snapshot<MujocoCommand> commands_;
  std::array<std::atomic<int>, 2> hand_commands_{{21, 21}};
  std::atomic<uint64_t> received_{0}, applied_{0}, rejected_{0}, samples_{0};
  std::atomic<bool> running_{false}, conflict_{false};
  std::thread worker_;
  rclcpp::Publisher<sensor_msgs::msg::JointState>::SharedPtr pub_;
  rclcpp::Publisher<std_msgs::msg::String>::SharedPtr status_;
  rclcpp::Subscription<std_msgs::msg::Float64MultiArray>::SharedPtr sub_;
  std::array<rclcpp::Subscription<std_msgs::msg::Int32>::SharedPtr, 2> hands_;
  rclcpp::TimerBase::SharedPtr report_;
};
} // namespace fa
int main(int argc, char **argv) {
  rclcpp::init(argc, argv);
  try {
    rclcpp::spin(std::make_shared<fa::MujocoNode>());
  } catch (const std::exception &e) {
    std::cerr << e.what() << '\n';
    rclcpp::shutdown();
    return 1;
  }
  rclcpp::shutdown();
  return 0;
}
