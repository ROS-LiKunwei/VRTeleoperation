#pragma once
#include "fa_teleop/core.hpp"
#include <memory>
#include <mujoco/mujoco.h>

namespace fa {
// Fixed-base kinematic mirror, matching the existing Python mj_forward backend.
// No IK, trajectory planning, interpolation, or free dynamics occurs here.
class MujocoEngine {
public:
  explicit MujocoEngine(const std::string &bundle);
  bool apply(const Upper &q);
  void hand(bool left, int ros_command);
  void forward(double elapsed_s);
  Upper positions() const;
  Upper velocities() const;
  Pose pose(bool left) const;
  mjModel *model() const { return model_.get(); }
  mjData *data() const { return data_.get(); }
  static void verify_runtime();

private:
  std::unique_ptr<mjModel, decltype(&mj_deleteModel)> model_{nullptr,
                                                             mj_deleteModel};
  std::unique_ptr<mjData, decltype(&mj_deleteData)> data_{nullptr,
                                                          mj_deleteData};
  std::array<int, 16> qpos_{}, dof_{};
  std::array<std::vector<int>, 2> hand_joints_;
  std::array<int, 2> endeff_{};
  int pelvis_{-1};
  Upper previous_{Upper::Zero()};
};
} // namespace fa
