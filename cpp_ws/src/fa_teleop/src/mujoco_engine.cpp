#include "fa_teleop/mujoco_engine.hpp"
#include "bundle_version.hpp"
#include <algorithm>
#include <dlfcn.h>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <regex>
#include <stdexcept>

namespace fa {
void MujocoEngine::verify_runtime() {
  if (mj_version() != mjVERSION_HEADER ||
      std::string(mj_versionString()) != kMujocoVersion)
    throw std::runtime_error("MuJoCo header/library version mismatch");
  Dl_info info{};
  if (!dladdr(reinterpret_cast<void *>(&mj_version), &info) ||
      !info.dli_fname || file_sha256(info.dli_fname) != kMujocoLibrarySha)
    throw std::runtime_error(
        "MuJoCo loaded runtime SHA does not match this build");
}
MujocoEngine::MujocoEngine(const std::string &bundle) {
  verify_runtime();
  const auto root =
      std::filesystem::canonical(bundle + "/models/fa_description");
  std::ifstream file(root / "urdf/fa_robot.urdf");
  std::string xml((std::istreambuf_iterator<char>(file)),
                  std::istreambuf_iterator<char>());
  if (xml.empty())
    throw std::runtime_error("Empty FA URDF");
  // Keep model bytes untouched on disk. The in-memory adapter only resolves
  // package URIs and preserves fixed TCP bodies / visual meshes in MuJoCo's
  // URDF importer.
  xml = std::regex_replace(
      xml, std::regex("package://(?:sysmo_description|fa_description)/meshes/"),
      (root / "meshes").string() + "/");
  const auto insertion = xml.find('>', xml.find("<robot"));
  if (insertion == std::string::npos)
    throw std::runtime_error("Invalid FA URDF root");
  xml.insert(insertion + 1,
             "\n<mujoco><compiler discardvisual=\"false\" fusestatic=\"false\" "
             "strippath=\"false\"/></mujoco>\n");
  mjVFS vfs;
  mj_defaultVFS(&vfs);
  const int added = mj_addBufferVFS(&vfs, "fa_robot.urdf", xml.data(),
                                    static_cast<int>(xml.size()));
  if (added) {
    mj_deleteVFS(&vfs);
    throw std::runtime_error("Cannot add FA URDF to MuJoCo VFS");
  }
  std::array<char, 2048> error{};
  model_.reset(mj_loadXML("fa_robot.urdf", &vfs, error.data(), error.size()));
  mj_deleteVFS(&vfs);
  if (!model_)
    throw std::runtime_error("MuJoCo model load: " + std::string(error.data()));
  // This is the original kinematic mirror, not a contact/force simulation.
  model_->opt.disableflags |= mjDSBL_CONTACT;
  model_->opt.timestep = 0.001;
  data_.reset(mj_makeData(model_.get()));
  if (!data_)
    throw std::runtime_error("MuJoCo mjData allocation failed");
  for (size_t i = 0; i < 16; ++i) {
    int joint = mj_name2id(model_.get(), mjOBJ_JOINT, joint_names()[i].c_str());
    if (joint < 0 || model_->jnt_type[joint] != mjJNT_HINGE)
      throw std::runtime_error("Missing scalar MuJoCo joint: " +
                               joint_names()[i]);
    qpos_[i] = model_->jnt_qposadr[joint];
    dof_[i] = model_->jnt_dofadr[joint];
    if (std::abs(model_->jnt_range[2 * joint] - lower_limits()[i]) > 0.002 ||
        std::abs(model_->jnt_range[2 * joint + 1] - upper_limits()[i]) > 0.002)
      throw std::runtime_error("MuJoCo model joint limits differ: " +
                               joint_names()[i]);
  }
  pelvis_ = mj_name2id(model_.get(), mjOBJ_BODY, "pelvis");
  endeff_ = {mj_name2id(model_.get(), mjOBJ_BODY, "left_hand_base_link"),
             mj_name2id(model_.get(), mjOBJ_BODY, "right_hand_base_link")};
  if (pelvis_ < 0 || endeff_[0] < 0 || endeff_[1] < 0)
    throw std::runtime_error("MuJoCo pelvis/TCP bodies missing");
  for (int j = 0; j < model_->njnt; ++j) {
    const char *raw = mj_id2name(model_.get(), mjOBJ_JOINT, j);
    if (!raw)
      continue;
    const std::string name = raw;
    if (name.find("thumb_") != std::string::npos ||
        name.find("index_") != std::string::npos ||
        name.find("middle_") != std::string::npos ||
        name.find("ring_") != std::string::npos ||
        name.find("pinky_") != std::string::npos) {
      if (name.rfind("left_", 0) == 0)
        hand_joints_[0].push_back(j);
      if (name.rfind("right_", 0) == 0)
        hand_joints_[1].push_back(j);
    }
  }
  // Neutral startup mirrors Python: explicit enable triggers the shared
  // initialization trajectory.
  apply(Upper::Zero());
  forward(0.001);
}
bool MujocoEngine::apply(const Upper &q) {
  if (!q.allFinite() || (q.array() < lower_limits().array()).any() ||
      (q.array() > upper_limits().array()).any())
    return false;
  for (size_t i = 0; i < 16; ++i)
    data_->qpos[qpos_[i]] = q[i];
  return true;
}
void MujocoEngine::hand(bool left, int command) {
  if (command != 20 && command != 21 && command != 22)
    return;
  for (int j : hand_joints_[left ? 0 : 1]) {
    const double lo = model_->jnt_range[2 * j],
                 hi = model_->jnt_range[2 * j + 1];
    // Finger pose is a display of the open/close command, not a contact/grasp
    // controller.
    data_->qpos[model_->jnt_qposadr[j]] =
        command == 20 ? lo + 0.65 * (hi - lo) : std::clamp(0.0, lo, hi);
  }
}
void MujocoEngine::forward(double elapsed) {
  const Upper q = positions();
  elapsed = std::clamp(elapsed, 0.0001, 0.05);
  for (size_t i = 0; i < 16; ++i)
    data_->qvel[dof_[i]] = (q[i] - previous_[i]) / elapsed;
  previous_ = q;
  data_->time += elapsed;
  mj_forward(model_.get(), data_.get());
}
Upper MujocoEngine::positions() const {
  Upper q;
  for (size_t i = 0; i < 16; ++i)
    q[i] = data_->qpos[qpos_[i]];
  return q;
}
Upper MujocoEngine::velocities() const {
  Upper q;
  for (size_t i = 0; i < 16; ++i)
    q[i] = data_->qvel[dof_[i]];
  return q;
}
Pose MujocoEngine::pose(bool left) const {
  const int ee = endeff_[left ? 0 : 1];
  const Eigen::Map<const Eigen::Matrix<double, 3, 3, Eigen::RowMajor>> base_R(
      data_->xmat + 9 * pelvis_),
      ee_R(data_->xmat + 9 * ee);
  const Eigen::Map<const Eigen::Vector3d> base_p(data_->xpos + 3 * pelvis_),
      ee_p(data_->xpos + 3 * ee);
  return {base_R.transpose() * (ee_p - base_p), base_R.transpose() * ee_R};
}
} // namespace fa
