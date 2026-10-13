#include "fa_teleop/msg/joint_target.hpp"
#include "fa_teleop/ros_helpers.hpp"
#include "fa_teleop/transport.hpp"
#include <sstream>
#include <std_msgs/msg/int32.hpp>
#include <std_msgs/msg/string.hpp>
#include <std_srvs/srv/trigger.hpp>

namespace fa {
enum class Stage {
  WaitingEnable,
  Initializing,
  Calibrating,
  Tracking,
  Paused,
  Fault
};
class TeleopNode final : public rclcpp::Node {
public:
  TeleopNode() : Node("fa_teleop") {
    bundle_ = bundle_directory(*this);
    config_ = node_config(*this, bundle_);
    enable_ = declare_parameter<bool>("enable_motion", false);
    kin_ = std::make_unique<Kinematics>(
        bundle_ + "/models/fa_description/urdf/fa_robot.urdf",
        bundle_ + "/models/fa_robot.srdf", config_);
    calibration_ = std::make_unique<AxisCalibration>(config_);
    vr_ = std::make_unique<VrTransport>(config_);
    target_ = create_publisher<fa_teleop::msg::JointTarget>(
        "fa_cpp/target", rclcpp::QoS(1).reliable());
    hands_[0] = create_publisher<std_msgs::msg::Int32>(
        "left_topic_to_hand", rclcpp::QoS(10).reliable());
    hands_[1] = create_publisher<std_msgs::msg::Int32>(
        "right_topic_to_hand", rclcpp::QoS(10).reliable());
    diagnostic_ =
        create_publisher<std_msgs::msg::String>("fa_cpp/teleop_status", 10);
    feedback_sub_ = create_subscription<sensor_msgs::msg::JointState>(
        "joint_states", rclcpp::SensorDataQoS().keep_last(1),
        [this](sensor_msgs::msg::JointState::ConstSharedPtr msg) {
          if (receive_feedback(*msg, fb_callback_))
            feedback_.write(fb_callback_);
        });
    executor_sub_ = create_subscription<std_msgs::msg::String>(
        "fa_cpp/executor_status", rclcpp::QoS(1).reliable(),
        [this](std_msgs::msg::String::ConstSharedPtr msg) {
          const auto &s = msg->data;
          const std::string prefix = deployment_id(config_) + "|";
          if (s.rfind(prefix, 0) != 0) {
            executor_fault_ = true;
            return;
          }
          if (s.find("|FAULT") != std::string::npos) {
            executor_fault_ = true;
            return;
          }
          executor_seen_.store(steady_seconds());
        });
    reset_ = create_service<std_srvs::srv::Trigger>(
        "fa_cpp/reset",
        [this](std_srvs::srv::Trigger::Request::SharedPtr,
               std_srvs::srv::Trigger::Response::SharedPtr response) {
          reset_requested_ = true;
          response->success = true;
          response->message =
              "Reset queued; explicit PICO enable still required";
        });
    vr_->start();
    running_ = true;
    worker_ = std::thread(&TeleopNode::run, this);
    RCLCPP_INFO(get_logger(), "FA C++ release=%s enable_motion=%s",
                release_id(), enable_ ? "true" : "false");
  }
  ~TeleopNode() override {
    running_ = false;
    if (worker_.joinable())
      worker_.join();
    vr_->stop();
  }

private:
  void send(uint8_t mode, const Upper &q, double duration, double velocity) {
    if (!enable_)
      return;
    fa_teleop::msg::JointTarget m;
    m.header.stamp = now();
    m.header.frame_id = "pelvis";
    m.release_id = deployment_id(config_);
    m.sequence = ++sequence_;
    m.mode = mode;
    for (int i = 0; i < 7; ++i) {
      m.left[i] = q[i];
      m.right[i] = q[i + 7];
    }
    m.neck[0] = q[14];
    m.neck[1] = q[15];
    m.duration_s = duration;
    m.velocity_limit = velocity;
    m.acceleration_limit = config_.acceleration_limit;
    target_->publish(m);
  }
  void pause(bool fault = false, const char *reason = "user_pause") {
    if (stage_ == Stage::Fault && !fault)
      return;
    stage_ = fault ? Stage::Fault : Stage::Paused;
    pause_reason_ = reason;
    last_hand_ = {0, 0};
  }
  void run() {
    Periodic schedule(config_.control_hz);
    HandSample l, r;
    Feedback fb;
    Upper command = initial_pose();
    bool initialized = false;
    double init_since = 0, settled_since = 0, last_report = 0, max_work = 0,
           max_late = 0;
    uint64_t cycles = 0, missed = 0, valid_solves = 0;
    std::array<Pose, 2> baseline{}, robot_baseline{}, last_pose{};
    uint64_t last_l = 0, last_r = 0;
    int jump_count = 0;
    while (running_ && rclcpp::ok()) {
      const double started = steady_seconds();
      feedback_.read(fb);
      vr_->left.read(l);
      vr_->right.read(r);
      // Samples may arrive after loop start but before their snapshots are
      // read.
      const double observed = steady_seconds();
      if (reset_requested_.exchange(false)) {
        stage_ = Stage::WaitingEnable;
        initialized = false;
        executor_fault_ = false;
        calibration_->reset();
        pause_reason_ = "waiting_enable";
      }
      if (vr_->fault || executor_fault_)
        pause(true, "transport_or_executor_fault");
      const bool feedback_ok = feedback_fresh(fb, config_);
      const bool hands_ok =
          l.valid && r.valid &&
          fresh(l.received, l.source_wall, observed, wall_seconds(),
                config_.input_timeout, config_) &&
          fresh(r.received, r.source_wall, observed, wall_seconds(),
                config_.input_timeout, config_) &&
          std::abs(l.received - r.received) <= config_.timestamp_skew;
      const bool executor_ok = started - executor_seen_.load() < 0.5;
      Event event;
      for (int n = 0; n < 64 && vr_->events.pop(event); ++n) {
        if (event == Event::Pause)
          pause();
        else if (event == Event::Resume && stage_ != Stage::Fault &&
                 feedback_ok && executor_ok && hands_ok && enable_) {
          command = fb.q;
          pause_reason_ = "none";
          calibration_->reset();
          last_hand_ = {0, 0};
          if (!initialized) {
            stage_ = Stage::Initializing;
            init_since = started;
            settled_since = 0;
          } else
            stage_ = Stage::Calibrating;
        }
      }
      if ((stage_ == Stage::Tracking || stage_ == Stage::Calibrating) &&
          (!hands_ok || !feedback_ok || !executor_ok))
        pause(false, !hands_ok ? "input_stale_or_unsynchronized"
                               : (!feedback_ok ? "feedback_stale"
                                               : "executor_heartbeat_lost"));
      if (stage_ == Stage::Initializing && (!feedback_ok || !executor_ok))
        pause(false,
              !feedback_ok ? "feedback_stale" : "executor_heartbeat_lost");
      if (stage_ == Stage::Initializing) {
        Upper goal = initial_pose();
        goal.tail<2>() = command.tail<2>();
        send(1, goal, config_.initialization_duration,
             config_.initialization_velocity);
        if ((fb.q.head<14>() - goal.head<14>()).cwiseAbs().maxCoeff() < 0.01) {
          if (settled_since == 0)
            settled_since = started;
          if (started - settled_since > 0.2) {
            initialized = true;
            command = fb.q;
            stage_ = Stage::Calibrating;
          }
        } else
          settled_since = 0;
        if (started - init_since > config_.initialization_duration + 10)
          pause(true, "initialization_timeout");
      } else if (stage_ == Stage::Calibrating) {
        send(0, command, config_.duration, config_.velocity_limit);
        if (!config_.calibration_enabled ||
            calibration_->update(l, r, started)) {
          baseline = {l.pose, r.pose};
          robot_baseline = {kin_->fk(fb.q.head<7>(), true),
                            kin_->fk(fb.q.segment<7>(7), false)};
          command = fb.q;
          last_pose = {l.pose, r.pose};
          last_l = l.sequence;
          last_r = r.sequence;
          jump_count = 0;
          stage_ = Stage::Tracking;
        }
        const int state = static_cast<int>(calibration_->state());
        if (state != last_calibration_) {
          vr_->prompts.push(state);
          last_calibration_ = state;
        }
        if (calibration_->state() == CalibrationState::Failed)
          pause(false, "calibration_failed");
      } else if (stage_ == Stage::Tracking) {
        if (l.sequence != last_l || r.sequence != last_r) {
          const Eigen::Vector3d dl = l.pose.p - last_pose[0].p,
                                dr = r.pose.p - last_pose[1].p;
          const bool common_jump =
              (dl + dr).norm() / 2 > config_.origin_jump_m &&
              (dl - dr).norm() < 0.04;
          const bool rotation_jump = rotation_error(l.pose.R, last_pose[0].R) >
                                         config_.origin_jump_rad &&
                                     rotation_error(r.pose.R, last_pose[1].R) >
                                         config_.origin_jump_rad;
          if (common_jump || rotation_jump) {
            if (++jump_count >= 2)
              pause(false, "tracking_origin_jump");
          } else {
            jump_count = 0;
            last_pose = {l.pose, r.pose};
          }
          last_l = l.sequence;
          last_r = r.sequence;
        }
        if (stage_ == Stage::Tracking && jump_count == 0) {
          const auto mapping = config_.calibration_enabled
                                   ? calibration_->mapping()
                                   : nominal_mapping();
          const std::array<Pose, 2> targets{
              retarget(l.pose, baseline[0], robot_baseline[0], mapping,
                       config_.translation_scale),
              retarget(r.pose, baseline[1], robot_baseline[1], mapping,
                       config_.translation_scale)};
          auto result = kin_->solve(targets, command);
          if (result.valid && safe_target(result.q, command, fb, config_)) {
            command = result.q;
            send(2, command, config_.duration, config_.velocity_limit);
            ++valid_solves;
            for (int i = 0; i < 2; ++i) {
              const int hand = i == 0 ? l.command : r.command;
              if (hand && hand != last_hand_[i]) {
                std_msgs::msg::Int32 m;
                m.data = hand == 1 ? config_.hand_open : config_.hand_close;
                hands_[i]->publish(m);
                last_hand_[i] = hand;
              }
            }
          } else
            pause(false, result.deadline_missed
                             ? "ik_deadline"
                             : (!result.valid ? "ik_precision_or_reachability"
                                              : "joint_safety_limit"));
        }
      }
      if ((stage_ == Stage::Paused || stage_ == Stage::Fault) && feedback_ok)
        send(0, command, config_.duration, config_.velocity_limit);
      const double work = steady_seconds() - started;
      max_work = std::max(max_work, work);
      ++cycles;
      if (work > 1 / config_.control_hz)
        ++missed;
      if (started - last_report >= 1) {
        std_msgs::msg::String m;
        std::ostringstream s;
        s << "{\"release_id\":\"" << release_id()
          << "\",\"stage\":" << static_cast<int>(stage_)
          << ",\"pause_reason\":\"" << pause_reason_ << "\""
          << ",\"cycles\":" << cycles << ",\"valid_solves\":" << valid_solves
          << ",\"compute_overruns\":" << missed
          << ",\"max_work_ms\":" << max_work * 1000
          << ",\"max_wakeup_ms\":" << max_late * 1000
          << ",\"vr_frames\":" << vr_->received_frames.load() << "}";
        m.data = s.str();
        diagnostic_->publish(m);
        last_report = started;
      }
      max_late = std::max(max_late, schedule.wait());
    }
    // The executor independently detects heartbeat loss and brakes the existing
    // reference.
  }
  std::string bundle_;
  Config config_;
  bool enable_{false};
  std::unique_ptr<Kinematics> kin_;
  std::unique_ptr<AxisCalibration> calibration_;
  std::unique_ptr<VrTransport> vr_;
  Snapshot<Feedback> feedback_;
  Feedback fb_callback_;
  Stage stage_{Stage::WaitingEnable};
  const char *pause_reason_{"waiting_enable"};
  std::array<int, 2> last_hand_{};
  int last_calibration_{-1};
  uint64_t sequence_{0};
  std::atomic<bool> running_{false}, executor_fault_{false},
      reset_requested_{false};
  std::atomic<double> executor_seen_{0};
  std::thread worker_;
  rclcpp::Publisher<fa_teleop::msg::JointTarget>::SharedPtr target_;
  std::array<rclcpp::Publisher<std_msgs::msg::Int32>::SharedPtr, 2> hands_;
  rclcpp::Publisher<std_msgs::msg::String>::SharedPtr diagnostic_;
  rclcpp::Subscription<sensor_msgs::msg::JointState>::SharedPtr feedback_sub_;
  rclcpp::Subscription<std_msgs::msg::String>::SharedPtr executor_sub_;
  rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr reset_;
};
} // namespace fa
int main(int argc, char **argv) {
  rclcpp::init(argc, argv);
  try {
    auto node = std::make_shared<fa::TeleopNode>();
    rclcpp::spin(node);
  } catch (const std::exception &e) {
    std::cerr << e.what() << '\n';
    rclcpp::shutdown();
    return 1;
  }
  rclcpp::shutdown();
  return 0;
}
