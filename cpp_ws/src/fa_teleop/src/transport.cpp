#include "fa_teleop/transport.hpp"
#include <cstring>
#include <iostream>
#include <zmq.hpp>
namespace fa {
VrTransport::VrTransport(const Config &c) : config_(c) {}
VrTransport::~VrTransport() { stop(); }
void VrTransport::start() {
  running_ = true;
  worker_ = std::thread(&VrTransport::run, this);
}
void VrTransport::stop() {
  running_ = false;
  if (worker_.joinable())
    worker_.join();
}
void VrTransport::run() {
  try {
    zmq::context_t context(1);
    zmq::socket_t left_socket(context, zmq::socket_type::pull),
        right_socket(context, zmq::socket_type::pull);
    zmq::socket_t session(context, zmq::socket_type::pull),
        button(context, zmq::socket_type::pull),
        prompt(context, zmq::socket_type::pub);
    std::array<zmq::socket_t *, 5> sockets{&left_socket, &right_socket,
                                           &session, &button, &prompt};
    std::array<int, 5> ports{config_.left_port, config_.right_port,
                             config_.event_port, config_.button_port,
                             config_.prompt_port};
    for (size_t i = 0; i < sockets.size(); ++i) {
      sockets[i]->set(zmq::sockopt::linger, 0);
      sockets[i]->set(zmq::sockopt::rcvhwm, i < 2 ? 1 : 64);
      sockets[i]->set(zmq::sockopt::maxmsgsize, int64_t(4096));
      if (i < 2)
        sockets[i]->set(zmq::sockopt::conflate, 1);
      sockets[i]->bind("tcp://" + config_.bind_address + ":" +
                       std::to_string(ports[i]));
    }
    prompt.set(zmq::sockopt::sndhwm, 5);
    uint64_t left_seq = 0, right_seq = 0;
    double last_resume = -1;
    std::string last_event;
    const std::array<const char *, 7> prompt_keys{
        "fa_calib_origin", "fa_calib_forward",       "fa_calib_return_origin",
        "fa_calib_left",   "fa_calib_return_origin", "fa_calib_success",
        "fa_calib_reset"};
    while (running_) {
      std::array<zmq::pollitem_t, 4> poll{
          {{left_socket.handle(), 0, ZMQ_POLLIN, 0},
           {right_socket.handle(), 0, ZMQ_POLLIN, 0},
           {session.handle(), 0, ZMQ_POLLIN, 0},
           {button.handle(), 0, ZMQ_POLLIN, 0}}};
      zmq::poll(poll, std::chrono::milliseconds(2));
      for (size_t i = 0; i < 4; ++i)
        if (poll[i].revents & ZMQ_POLLIN) {
          // Bound work per poll cycle. Session events preserve order, unlike
          // pose conflation.
          for (int n = 0; n < (i < 2 ? 1 : 64); ++n) {
            zmq::message_t msg;
            const auto got = sockets[i]->recv(msg, zmq::recv_flags::dontwait);
            if (!got)
              break;
            const std::string raw = msg.to_string();
            const double now = steady_seconds();
            if (i < 2) {
              HandSample sample;
              auto &seq = i == 0 ? left_seq : right_seq;
              if (parse_hand(raw, i == 0, now, wall_seconds(), ++seq, sample)) {
                (i == 0 ? left : right).write(sample);
                ++received_frames;
              } else
                ++invalid_frames;
            } else if (i == 2) {
              if (raw == "KEEPALIVE")
                continue;
              if (raw != "High" && raw != "Low")
                continue;
              if (raw == "Low" && now - last_resume < 0.1)
                continue;
              if (raw == last_event)
                continue;
              last_event = raw;
              if (raw == "High")
                last_resume = now;
              if (!events.push(raw == "High" ? Event::Resume : Event::Pause))
                fault = true;
            } else if (raw == "High" || raw == "Low") {
              if (!events.push(raw == "High" ? Event::HighResolution
                                             : Event::LowResolution))
                fault = true;
            }
          }
        }
      int state;
      while (prompts.pop(state))
        if (state >= 0 && state < 7) {
          const auto topic = zmq::str_buffer("fa_calibration_prompt");
          prompt.send(topic,
                      zmq::send_flags::sndmore | zmq::send_flags::dontwait);
          prompt.send(
              zmq::buffer(prompt_keys[state], std::strlen(prompt_keys[state])),
              zmq::send_flags::dontwait);
        }
    }
  } catch (const std::exception &e) {
    fault = true;
    std::cerr << "VR transport failed: " << e.what() << '\n';
  }
}
} // namespace fa
