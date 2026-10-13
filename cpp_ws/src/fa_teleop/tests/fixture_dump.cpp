#include "fa_teleop/core.hpp"
#include <iomanip>
#include <iostream>
template <class T> void vector_json(const T &value) {
  std::cout << '[';
  for (int i = 0; i < value.size(); ++i) {
    if (i)
      std::cout << ',';
    std::cout << value[i];
  }
  std::cout << ']';
}
void matrix_json(const Eigen::Matrix3d &m) {
  std::cout << '[';
  for (int row = 0; row < 3; ++row) {
    if (row)
      std::cout << ',';
    vector_json(m.row(row).transpose().eval());
  }
  std::cout << ']';
}
int main() {
  std::cout << std::setprecision(17) << '[';
  for (int side = 0; side < 2; ++side) {
    if (side)
      std::cout << ',';
    std::array<Eigen::Vector3d, 26> points;
    for (auto &p : points)
      p = Eigen::Vector3d(0.2, 0.1, 0.3);
    const Eigen::Matrix3d r =
        Eigen::AngleAxisd(0.4, Eigen::Vector3d(1, 2, 3).normalized())
            .toRotationMatrix();
    points[7] += r * Eigen::Vector3d(0.03, 0.08, 0);
    points[12] += r * Eigen::Vector3d(0, 0.09, 0);
    points[22] += r * Eigen::Vector3d(-0.03, 0.07, 0);
    fa::Pose pose;
    if (!fa::valid_hand_pose(points, side == 0, pose))
      return 1;
    fa::Pose baseline = pose, robot;
    baseline.p -= Eigen::Vector3d(0.01, 0.02, 0.03);
    baseline.R =
        Eigen::AngleAxisd(-0.2, Eigen::Vector3d::UnitZ()).toRotationMatrix() *
        pose.R;
    robot.p = {0.4, 0.2, 0.5};
    robot.R =
        Eigen::AngleAxisd(0.3, Eigen::Vector3d::UnitX()).toRotationMatrix();
    const auto target =
        fa::retarget(pose, baseline, robot, fa::nominal_mapping(), 1);
    std::cout << "{\"left\":" << (side == 0 ? "true" : "false")
              << ",\"points\":[";
    for (size_t i = 0; i < 26; ++i) {
      if (i)
        std::cout << ',';
      vector_json(points[i]);
    }
    std::cout << "],\"R\":";
    matrix_json(pose.R);
    std::cout << ",\"baseline_p\":";
    vector_json(baseline.p);
    std::cout << ",\"baseline_R\":";
    matrix_json(baseline.R);
    std::cout << ",\"robot_p\":";
    vector_json(robot.p);
    std::cout << ",\"robot_R\":";
    matrix_json(robot.R);
    std::cout << ",\"target_p\":";
    vector_json(target.p);
    std::cout << ",\"target_R\":";
    matrix_json(target.R);
    std::cout << '}';
  }
  std::cout << "]\n";
}
