#pragma once
#include "fa_teleop/core.hpp"
#include <thread>
namespace fa {
enum class Event { Resume, Pause, HighResolution, LowResolution };
class VrTransport {
public:
  explicit VrTransport(const Config &c);
  ~VrTransport();
  void start();
  void stop();
  Snapshot<HandSample> left, right;
  SpscQueue<Event, 64> events;
  SpscQueue<int, 64> prompts;
  std::atomic<bool> fault{false};
  std::atomic<uint64_t> invalid_frames{0}, received_frames{0};

private:
  void run();
  Config config_;
  std::atomic<bool> running_{false};
  std::thread worker_;
};
} // namespace fa
