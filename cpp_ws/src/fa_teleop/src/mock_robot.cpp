#include "fa_teleop/ros_helpers.hpp"
#include <iostream>
#include <std_msgs/msg/float64_multi_array.hpp>
// Kinematic test plant only. Namespaced launch required; never drives physical
// hardware.
class MockRobot : public rclcpp::Node {
public:
  MockRobot() : Node("fa_mock_robot") {
    if (get_namespace() == std::string("/"))
      throw std::runtime_error("Mock robot requires an isolated ROS namespace");
    fa::bundle_directory(*this);
    q_ = fa::initial_pose();
    target_ = q_;
    pub_ = create_publisher<sensor_msgs::msg::JointState>(
        "joint_states", rclcpp::SensorDataQoS().keep_last(1));
    sub_ = create_subscription<std_msgs::msg::Float64MultiArray>(
        "upper_position_controller/commands", rclcpp::QoS(1).reliable(),
        [this](std_msgs::msg::Float64MultiArray::ConstSharedPtr m) {
          if (m->data.size() != 16)
            return;
          for (int i = 0; i < 16; ++i)
            if (!std::isfinite(m->data[i]))
              return;
          for (int i = 0; i < 16; ++i)
            target_[i] = m->data[i];
        });
    timer_ = create_wall_timer(std::chrono::milliseconds(1), [this] {
      const auto old = q_;
      q_ += 0.25 * (target_ - q_);
      sensor_msgs::msg::JointState m;
      m.header.stamp = now();
      m.name.assign(fa::joint_names().begin(), fa::joint_names().end());
      m.position.assign(q_.data(), q_.data() + 16);
      const fa::Upper v = (q_ - old) / 0.001;
      m.velocity.assign(v.data(), v.data() + 16);
      pub_->publish(m);
    });
  }

private:
  fa::Upper q_, target_;
  rclcpp::Publisher<sensor_msgs::msg::JointState>::SharedPtr pub_;
  rclcpp::Subscription<std_msgs::msg::Float64MultiArray>::SharedPtr sub_;
  rclcpp::TimerBase::SharedPtr timer_;
};
int main(int argc, char **argv) {
  rclcpp::init(argc, argv);
  try {
    rclcpp::spin(std::make_shared<MockRobot>());
  } catch (const std::exception &e) {
    std::cerr << e.what() << '\n';
    rclcpp::shutdown();
    return 1;
  }
  rclcpp::shutdown();
}
