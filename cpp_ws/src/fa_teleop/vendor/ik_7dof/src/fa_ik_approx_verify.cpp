#include "ik_7dof/fa_ik_solver.hpp"

#include <Eigen/Dense>
#include <cmath>
#include <iostream>
#include <string>

namespace
{

bool isFiniteVector(const Eigen::VectorXd& q)
{
    for (int i = 0; i < q.size(); ++i) {
        if (!std::isfinite(q[i])) {
            return false;
        }
    }
    return true;
}

bool isWithinLimits(
    const Eigen::VectorXd& q,
    const std::pair<Eigen::VectorXd, Eigen::VectorXd>& limits)
{
    for (int i = 0; i < q.size(); ++i) {
        if (q[i] < limits.first[i] - 1e-9 || q[i] > limits.second[i] + 1e-9) {
            return false;
        }
    }
    return true;
}

}  // namespace

int main()
{
    using namespace fa_arm_kinematic;

    const std::string urdf_file =
        std::string(IK_7DOF_WORKSPACE_SRC_DIR) + "/sysmo_description/urdf/fa_robot.urdf";
    const std::string srdf_file =
        std::string(IK_7DOF_WORKSPACE_SRC_DIR) + "/fa_moveit2_config/config/fa_robot.srdf";

    IKSolver solver(urdf_file, srdf_file);
    ArmKinematicsOptions options;
    options.reference_frame = ArmReferenceFrame::PELVIS;

    Eigen::VectorXd reachable_seed(7);
    reachable_seed << -0.68, 0.82, -0.97, -1.05, -0.36, 0.0, 0.15;
    const PoseSE3 seed_pose = solver.computeArmFK_SE3(
        reachable_seed, ArmSide::LEFT, options);
    pinocchio::SE3 reachable_target(seed_pose.R, seed_pose.p);
    reachable_target.translation().x() += 0.005;

    constexpr double convergence_eps = 1e-3;
    const IKResult reachable_result = solver.solveArmIK(
        reachable_target,
        ArmSide::LEFT,
        reachable_seed,
        options,
        50,
        convergence_eps);
    const double reachable_error_norm = std::hypot(
        reachable_result.position_error, reachable_result.orientation_error);
    if (!reachable_result.success || reachable_error_norm > convergence_eps + 1e-9) {
        std::cerr << "可达目标未满足 eps=1e-3 的末端误差停止条件: error_norm="
                  << reachable_error_norm << std::endl;
        return 1;
    }

    pinocchio::SE3 unreachable_target;
    unreachable_target.translation(Eigen::Vector3d(-0.00768, -0.32075, -0.053335));
    Eigen::Quaterniond target_quat(0.001, 0.707, 0.707, 0.001);
    target_quat.normalize();
    unreachable_target.rotation(target_quat.toRotationMatrix());

    IKResult ik_result = solver.solveArmIK(
        unreachable_target,
        ArmSide::RIGHT,
        Eigen::VectorXd(),
        options);

    const auto limits = solver.getArmJointLimits(ArmSide::RIGHT);
    if (!ik_result.has_solution) {
        std::cerr << "solveArmIK 对不可达目标没有返回可用近似解" << std::endl;
        return 1;
    }

    const Eigen::VectorXd& q = ik_result.q_solution;
    if (q.size() != 7) {
        std::cerr << "solveArmIK 对不可达目标返回了空解" << std::endl;
        return 1;
    }
    if (!isFiniteVector(q) || !isWithinLimits(q, limits)) {
        std::cerr << "solveArmIK 返回的近似解包含非法数值或超出关节限位" << std::endl;
        return 1;
    }
    if (q.cwiseAbs().maxCoeff() <= 1e-9) {
        std::cerr << "solveArmIK 对不可达目标返回了全 0 关节角" << std::endl;
        return 1;
    }

    const PoseSE3 approx_pose = solver.computeArmFK_SE3(q, ArmSide::RIGHT, options);
    const double position_error = (unreachable_target.translation() - approx_pose.p).norm();
    if (!std::isfinite(position_error)) {
        std::cerr << "近似解 FK 结果无效" << std::endl;
        return 1;
    }
    if (!std::isfinite(ik_result.position_error) || !std::isfinite(ik_result.orientation_error)) {
        std::cerr << "IKResult 误差字段无效" << std::endl;
        return 1;
    }

    const Eigen::Vector3d target_position = unreachable_target.translation();
    const Eigen::Quaterniond target_orientation(unreachable_target.rotation());
    std::cout << "目标位姿: position=[" << target_position.x() << ", "
              << target_position.y() << ", " << target_position.z()
              << "], quaternion(xyzw)=[" << target_orientation.x() << ", "
              << target_orientation.y() << ", " << target_orientation.z()
              << ", " << target_orientation.w() << "]" << std::endl;
    if (ik_result.success) {
        std::cout << "IK 精确可达" << std::endl;
    } else {
        std::cout << "IK 未达到精确阈值，返回近似解" << std::endl;
    }
    std::cout << "q_solution=[";
    for (int i = 0; i < q.size(); ++i) {
        std::cout << q[i] << (i + 1 < q.size() ? ", " : "");
    }
    std::cout << "]" << std::endl;
    std::cout << "position_error=" << position_error
              << " m, orientation_error=" << ik_result.orientation_error
              << " rad, iterations=" << ik_result.iterations << std::endl;
    return 0;
}
