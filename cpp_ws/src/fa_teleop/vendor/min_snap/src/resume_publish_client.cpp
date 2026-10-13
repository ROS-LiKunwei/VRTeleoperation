#include <chrono>
#include <memory>
#include <string>

#include "rclcpp/rclcpp.hpp"
#include "std_srvs/srv/trigger.hpp"

using namespace std::chrono_literals;

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);

  auto node = rclcpp::Node::make_shared("min_snap_resume_publish_client");
  const std::string service_name = argc > 1 ? argv[1] : "/min_snap/resume_trajectory_publish";
  auto client = node->create_client<std_srvs::srv::Trigger>(service_name);

  RCLCPP_INFO(node->get_logger(), "Waiting for service: %s", service_name.c_str());
  while (!client->wait_for_service(1s)) {
    if (!rclcpp::ok()) {
      RCLCPP_ERROR(node->get_logger(), "Interrupted while waiting for service.");
      rclcpp::shutdown();
      return 1;
    }
  }

  auto request = std::make_shared<std_srvs::srv::Trigger::Request>();
  auto future = client->async_send_request(request);
  if (rclcpp::spin_until_future_complete(node, future) != rclcpp::FutureReturnCode::SUCCESS) {
    RCLCPP_ERROR(node->get_logger(), "Failed to call service: %s", service_name.c_str());
    rclcpp::shutdown();
    return 1;
  }

  const auto response = future.get();
  if (response->success) {
    RCLCPP_INFO(node->get_logger(), "Resume request accepted: %s", response->message.c_str());
  } else {
    RCLCPP_ERROR(node->get_logger(), "Resume request rejected: %s", response->message.c_str());
  }

  rclcpp::shutdown();
  return response->success ? 0 : 2;
}
