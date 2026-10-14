#include "fa_teleop/mujoco_engine.hpp"
#include "fa_teleop/ros_helpers.hpp"
#include <GLFW/glfw3.h>
#include <fstream>
#include <iostream>
#include <std_msgs/msg/int32.hpp>

namespace fa {
struct CameraControls {
  mjvCamera *camera;
  const mjModel *model;
  mjvScene *scene;
  double x{0}, y{0};
};
class MujocoViewer final : public rclcpp::Node {
public:
  MujocoViewer() : Node("fa_mujoco_viewer") {
    const auto bundle = bundle_directory(*this);
    config_ = node_config(*this, bundle);
    if (config_.control_backend != "mujoco")
      throw std::runtime_error("Viewer requires control_backend:=mujoco");
    engine_ = std::make_unique<MujocoEngine>(bundle);
    render_hz_ = declare_parameter<double>("render_hz", 30);
    if (!std::isfinite(render_hz_) || render_hz_ < 1 || render_hz_ > 120)
      throw std::runtime_error("render_hz must be 1..120");
    max_frames_ = declare_parameter<int>("max_frames", 0);
    visible_ = declare_parameter<bool>("visible", true);
    screenshot_ = declare_parameter<std::string>("screenshot_path", "");
    sub_ = create_subscription<sensor_msgs::msg::JointState>(
        "joint_states", rclcpp::SensorDataQoS().keep_last(1),
        [this](sensor_msgs::msg::JointState::ConstSharedPtr m) {
          if (receive_feedback(*m, feedback_))
            received_ = true;
        });
    for (int i = 0; i < 2; ++i)
      hands_[i] = create_subscription<std_msgs::msg::Int32>(
          i == 0 ? "left_topic_to_hand" : "right_topic_to_hand", 10,
          [this, i](std_msgs::msg::Int32::ConstSharedPtr m) {
            hand_commands_[i] = m->data;
          });
  }
  void run() {
    if (!glfwInit())
      throw std::runtime_error(
          "GLFW initialization failed; use render:=false on a headless host");
    glfwWindowHint(GLFW_VISIBLE, visible_ ? GLFW_TRUE : GLFW_FALSE);
    GLFWwindow *window = glfwCreateWindow(
        1100, 850, "FA C++ MuJoCo — simulation", nullptr, nullptr);
    if (!window) {
      glfwTerminate();
      throw std::runtime_error("Cannot create MuJoCo OpenGL window");
    }
    glfwMakeContextCurrent(window);
    glfwSwapInterval(0);
    mjvCamera camera;
    mjv_defaultCamera(&camera);
    for (int i = 0; i < 3; ++i)
      camera.lookat[i] = engine_->model()->stat.center[i];
    camera.distance = std::max(3.2, 1.5 * engine_->model()->stat.extent);
    camera.azimuth = 120;
    camera.elevation = -15;
    mjvOption option;
    mjv_defaultOption(&option);
    option.geomgroup[0] = 0;
    option.geomgroup[1] = 1;
    mjvScene scene;
    mjv_defaultScene(&scene);
    mjv_makeScene(engine_->model(), &scene, 4000);
    mjrContext context;
    mjr_defaultContext(&context);
    mjr_makeContext(engine_->model(), &context, mjFONTSCALE_150);
    CameraControls controls{&camera, engine_->model(), &scene};
    glfwSetWindowUserPointer(window, &controls);
    glfwSetCursorPosCallback(window, [](GLFWwindow *w, double x, double y) {
      auto &c = *static_cast<CameraControls *>(glfwGetWindowUserPointer(w));
      int height = 1;
      glfwGetWindowSize(w, nullptr, &height);
      const double dx = x - c.x, dy = y - c.y;
      c.x = x;
      c.y = y;
      const bool left = glfwGetMouseButton(w, GLFW_MOUSE_BUTTON_LEFT) ==
                        GLFW_PRESS,
                 right = glfwGetMouseButton(w, GLFW_MOUSE_BUTTON_RIGHT) ==
                         GLFW_PRESS,
                 middle = glfwGetMouseButton(w, GLFW_MOUSE_BUTTON_MIDDLE) ==
                          GLFW_PRESS;
      if (!left && !right && !middle)
        return;
      const bool shift = glfwGetKey(w, GLFW_KEY_LEFT_SHIFT) == GLFW_PRESS ||
                         glfwGetKey(w, GLFW_KEY_RIGHT_SHIFT) == GLFW_PRESS;
      const int action =
          right ? (shift ? mjMOUSE_MOVE_H : mjMOUSE_MOVE_V)
                : (left ? (shift ? mjMOUSE_ROTATE_H : mjMOUSE_ROTATE_V)
                        : mjMOUSE_ZOOM);
      mjv_moveCamera(c.model, action, dx / std::max(1, height),
                     dy / std::max(1, height), c.scene, c.camera);
    });
    glfwSetScrollCallback(window, [](GLFWwindow *w, double, double y) {
      auto &c = *static_cast<CameraControls *>(glfwGetWindowUserPointer(w));
      mjv_moveCamera(c.model, mjMOUSE_ZOOM, 0, -0.05 * y, c.scene, c.camera);
    });
    glfwSetKeyCallback(window,
                       [](GLFWwindow *w, int key, int, int action, int) {
                         if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS)
                           glfwSetWindowShouldClose(w, GLFW_TRUE);
                       });
    Periodic clock(render_hz_);
    int frames = 0;
    while (rclcpp::ok() && !glfwWindowShouldClose(window) &&
           (max_frames_ <= 0 || frames < max_frames_)) {
      rclcpp::spin_some(shared_from_this());
      if (received_)
        engine_->apply(feedback_.q);
      else
        engine_->apply(initial_pose());
      engine_->hand(true, hand_commands_[0]);
      engine_->hand(false, hand_commands_[1]);
      engine_->forward(1 / render_hz_);
      mjrRect viewport{0, 0, 0, 0};
      glfwGetFramebufferSize(window, &viewport.width, &viewport.height);
      mjv_updateScene(engine_->model(), engine_->data(), &option, nullptr,
                      &camera, mjCAT_ALL, &scene);
      mjr_render(viewport, &scene, &context);
      if (!screenshot_.empty() && max_frames_ > 0 &&
          frames == max_frames_ - 1) {
        std::vector<unsigned char> rgb(static_cast<size_t>(viewport.width) *
                                       viewport.height * 3);
        mjr_readPixels(rgb.data(), nullptr, viewport, &context);
        std::ofstream image(screenshot_, std::ios::binary);
        image << "P6\n"
              << viewport.width << ' ' << viewport.height << "\n255\n";
        for (int row = viewport.height - 1; row >= 0; --row)
          image.write(
              reinterpret_cast<const char *>(
                  rgb.data() + static_cast<size_t>(row) * viewport.width * 3),
              viewport.width * 3);
        if (!image)
          throw std::runtime_error("Cannot save MuJoCo preview");
      }
      glfwSwapBuffers(window);
      glfwPollEvents();
      ++frames;
      clock.wait();
    }
    mjr_freeContext(&context);
    mjv_freeScene(&scene);
    glfwDestroyWindow(window);
    glfwTerminate();
    RCLCPP_INFO(get_logger(), "MuJoCo viewer closed after %d frames", frames);
  }

private:
  Config config_;
  std::unique_ptr<MujocoEngine> engine_;
  Feedback feedback_;
  bool received_{false}, visible_{true};
  double render_hz_{30};
  int max_frames_{0};
  std::string screenshot_;
  std::array<int, 2> hand_commands_{{21, 21}};
  rclcpp::Subscription<sensor_msgs::msg::JointState>::SharedPtr sub_;
  std::array<rclcpp::Subscription<std_msgs::msg::Int32>::SharedPtr, 2> hands_;
};
} // namespace fa
int main(int argc, char **argv) {
  rclcpp::init(argc, argv);
  try {
    auto viewer = std::make_shared<fa::MujocoViewer>();
    viewer->run();
  } catch (const std::exception &e) {
    std::cerr << e.what() << '\n';
    rclcpp::shutdown();
    return 1;
  }
  rclcpp::shutdown();
  return 0;
}
