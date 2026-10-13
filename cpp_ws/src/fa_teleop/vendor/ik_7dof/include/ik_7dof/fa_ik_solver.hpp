#ifndef FA_ARM_KINEMATIC__FA_IK_SOLVER_HPP_
#define FA_ARM_KINEMATIC__FA_IK_SOLVER_HPP_

#include <string>
#include <vector>
#include <array>
#include <memory>
#include <chrono>
#include <Eigen/Dense>
#include <pinocchio/spatial/se3.hpp>

namespace fa_arm_kinematic
{

struct PoseRPY
{
    double x, y, z;
    double roll, pitch, yaw;
};

struct PoseSE3 {
    Eigen::Vector3d p;
    Eigen::Matrix3d R;
};

enum class ArmReferenceFrame {
    ARM_BASE,
    PELVIS
};

struct ArmKinematicsOptions {
    // v1.0.0: cooperative deadline at iteration boundaries; disabled for legacy callers.
    std::chrono::steady_clock::time_point deadline = std::chrono::steady_clock::time_point::max();
    ArmReferenceFrame reference_frame = ArmReferenceFrame::PELVIS;
    bool skip_svd_fallback = false;
    double position_weight = 1.0;
    double orientation_weight = 1.0;
    double acceptable_position_error = 0.05;
    double acceptable_orientation_error = 0.05;
    double continuity_nullspace_weight = 0.0;
};

struct IKResult
{
    bool success = false;                 // 是否达到精确阈值
    bool has_solution = false;            // 是否有可用近似解
    Eigen::VectorXd q_solution;           // 返回的关节角
    double position_error = 0.0;          // 位置误差，单位 m
    double orientation_error = 0.0;       // 姿态误差，单位 rad
    int iterations = 0;                   // 实际迭代次数
};

enum class ArmSide {
    LEFT,
    RIGHT
};

enum class SolverMethod {
    LDLT,
    SVD
};

class IKSolver
{
public:
    IKSolver(const std::string& urdf_file, const std::string& srdf_file = "");
    ~IKSolver();

    void setComfortNullspace(
        double weight,
        const Eigen::VectorXd& left_q_ref,
        const Eigen::VectorXd& right_q_ref);

    PoseSE3 computeArmFK_SE3(const Eigen::VectorXd& q, ArmSide arm_side,
                             const ArmKinematicsOptions& options = ArmKinematicsOptions{});

    Eigen::VectorXd solveIK_Core(
        const pinocchio::SE3& T_target,
        const Eigen::VectorXd& q_init,
        int max_iters,
        // 未加权 log6(SE3) 末端误差范数的收敛阈值。
        double eps,
        int& iters_out,
        SolverMethod method,
        ArmSide arm_side = ArmSide::LEFT,
        const ArmKinematicsOptions& options = ArmKinematicsOptions{},
        bool* exact_solution = nullptr,
        bool* acceptable_solution = nullptr,
        double* best_error = nullptr);

    IKResult solveArmIK(
        const pinocchio::SE3 T_target_pose,
        const ArmSide arm_side,
        const Eigen::VectorXd initial_q,
        const ArmKinematicsOptions options,
        const int max_iters = 200,
        // 未加权 log6(SE3) 末端误差范数的收敛阈值。
        const double eps = 1e-3);

    PoseRPY computeArmFK(const Eigen::VectorXd& q, ArmSide arm_side,
                         const ArmKinematicsOptions& options = ArmKinematicsOptions{});

    const std::vector<std::string>& getArmJointNames(ArmSide arm_side) const;
    size_t getArmJointCount(ArmSide arm_side) const;

    std::pair<Eigen::VectorXd, Eigen::VectorXd> getArmJointLimits(ArmSide arm_side) const;

    struct IKStatistics {
        int total_calls = 0;
        int qr_stage_success = 0;
        int svd_stage_success = 0;
        int total_failures = 0;

        void reset() {
            total_calls = qr_stage_success = svd_stage_success = total_failures = 0;
        }
    };
    IKStatistics stats_;

private:
    class Impl;
    std::unique_ptr<Impl> pimpl_;

    // fa_robot 左臂 7 关节
    const std::vector<std::string> left_arm_joints_ = {
        "left_shoulder_pitch_joint",
        "left_shoulder_roll_joint",
        "left_shoulder_yaw_joint",
        "left_elbow_joint",
        "left_wrist_yaw_joint",
        "left_wrist_pitch_joint",
        "left_wrist_roll_joint"
    };

    // fa_robot 右臂 7 关节
    const std::vector<std::string> right_arm_joints_ = {
        "right_shoulder_pitch_joint",
        "right_shoulder_roll_joint",
        "right_shoulder_yaw_joint",
        "right_elbow_joint",
        "right_wrist_yaw_joint",
        "right_wrist_pitch_joint",
        "right_wrist_roll_joint"
    };

    mutable std::vector<int> q_indices_;
    mutable std::vector<int> v_indices_;

    // fa_robot 左手末端 frame
    const std::string left_ee_frame_ = "left_hand_base_link";
    // fa_robot 右手末端 frame
    const std::string right_ee_frame_ = "right_hand_base_link";
    const std::string pelvis_frame_ = "pelvis";
    const std::string left_arm_base_frame_ = "waist_pitch_link";
    const std::string right_arm_base_frame_ = "waist_pitch_link";
    const std::vector<std::string> waist_joints_ = {
        "waist_yaw_joint",
        "waist_roll_joint",
        "waist_pitch_joint"
    };

    std::string urdf_file_;
    std::string srdf_file_;
    double comfort_nullspace_weight_ = 0.08;
    Eigen::VectorXd comfort_left_q_ref_;
    Eigen::VectorXd comfort_right_q_ref_;
};

} // namespace fa_arm_kinematic

#endif // FA_ARM_KINEMATIC__FA_IK_SOLVER_HPP_
