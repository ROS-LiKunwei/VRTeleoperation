#include <rclcpp/rclcpp.hpp>
#include "ik_7dof/fa_ik_solver.hpp"
#include <Eigen/Dense>
#include <chrono>
#include <fstream>
#include <memory>
#include <random>
#include <string>
using namespace fa_arm_kinematic;

class FaArmKinematicNode : public rclcpp::Node
{
public:
    FaArmKinematicNode() : Node("fa_arm_kinematic_node")
    {
        this->declare_parameter<std::string>("urdf_file", "");
        this->declare_parameter<std::string>("srdf_file", "");
        this->declare_parameter<std::string>("arm_side", "left");
        this->declare_parameter<int>("num_tests", 10);
        this->declare_parameter<int>("max_iters", 200);
        this->declare_parameter<double>("eps", 1e-3);
        this->declare_parameter<std::string>("reference_frame", "pelvis");

        std::string urdf_file = this->get_parameter("urdf_file").as_string();
        std::string srdf_file = this->get_parameter("srdf_file").as_string();
        std::string arm_side_str = this->get_parameter("arm_side").as_string();
        std::string reference_frame = this->get_parameter("reference_frame").as_string();
        num_tests_ = this->get_parameter("num_tests").as_int();
        max_iters_ = this->get_parameter("max_iters").as_int();
        eps_ = this->get_parameter("eps").as_double();

        if (urdf_file.empty()) {
            RCLCPP_ERROR(this->get_logger(), "必须提供urdf_file参数");
            return;
        }

        if (arm_side_str == "right") {
            arm_side_ = ArmSide::RIGHT;
            arm_name_ = "右臂";
        } else {
            arm_side_ = ArmSide::LEFT;
            arm_name_ = "左臂";
        }

        if (reference_frame == "arm_base") {
            kinematics_options_.reference_frame = ArmReferenceFrame::ARM_BASE;
            reference_frame_name_ = "arm_base";
        } else {
            kinematics_options_.reference_frame = ArmReferenceFrame::PELVIS;
            reference_frame_name_ = "pelvis";
        }

        RCLCPP_INFO(this->get_logger(), "URDF文件: %s", urdf_file.c_str());
        RCLCPP_INFO(this->get_logger(), "SRDF文件: %s", srdf_file.c_str());
        RCLCPP_INFO(this->get_logger(), "手臂: %s", arm_name_.c_str());
        RCLCPP_INFO(this->get_logger(), "测试次数: %d", num_tests_);
        RCLCPP_INFO(this->get_logger(), "参考坐标系: %s", reference_frame_name_.c_str());

        try {
            solver_ = std::make_unique<IKSolver>(urdf_file, srdf_file);
            printJointInfo();
            runValidation();
        } catch (const std::exception& e) {
            RCLCPP_ERROR(this->get_logger(), "初始化失败: %s", e.what());
        }
    }

private:
    Eigen::VectorXd sampleRandomJointValues(
        const std::pair<Eigen::VectorXd, Eigen::VectorXd>& limits,
        const std::vector<std::string>& joint_names,
        std::mt19937& gen) const
    {
        const int joint_count = static_cast<int>(joint_names.size());
        Eigen::VectorXd q_rand(joint_count);
        for (int i = 0; i < joint_count; ++i) {
            if (limits.second[i] > limits.first[i] + 1e-6) {
                std::uniform_real_distribution<double> dist(limits.first[i], limits.second[i]);
                q_rand[i] = dist(gen);
            } else {
                q_rand[i] = 0.0;
                RCLCPP_WARN(this->get_logger(), "关节 %s 限制异常 [%.3f, %.3f]",
                    joint_names[i].c_str(), limits.first[i], limits.second[i]);
            }
        }
        return q_rand;
    }

    void logJointVector(const char* title,
                        const std::vector<std::string>& joint_names,
                        const Eigen::VectorXd& q) const
    {
        RCLCPP_INFO(this->get_logger(), "%s", title);
        for (int i = 0; i < q.size() && i < static_cast<int>(joint_names.size()); ++i) {
            RCLCPP_INFO(this->get_logger(), "  %s: %.6f rad", joint_names[i].c_str(), q[i]);
        }
    }

    pinocchio::SE3 makeTargetPose(const PoseSE3& pose) const
    {
        pinocchio::SE3 target;
        target.translation(pose.p);
        target.rotation(pose.R);
        return target;
    }

    void writeFailureCase(std::ofstream& log_file,
                          int test_index,
                          const pinocchio::SE3& target_pose,
                          const Eigen::VectorXd& q_rand) const
    {
        if (!log_file.is_open()) {
            return;
        }

        Eigen::Vector3d euler = target_pose.rotation().eulerAngles(2, 1, 0);
        log_file << test_index << ", "
                 << target_pose.translation().x() << ", "
                 << target_pose.translation().y() << ", "
                 << target_pose.translation().z() << ", "
                 << euler[0] << ", "
                 << euler[1] << ", "
                 << euler[2];
        for (int i = 0; i < q_rand.size(); ++i) {
            log_file << ", " << q_rand[i];
        }
        log_file << "\n";
    }

    void printJointInfo()
    {
        RCLCPP_INFO(this->get_logger(), "FA%s关节 (%zu 个):", arm_name_.c_str(), solver_->getArmJointCount(arm_side_));
        const auto& joint_names = solver_->getArmJointNames(arm_side_);
        for (size_t i = 0; i < joint_names.size(); ++i) {
            RCLCPP_INFO(this->get_logger(), "  %2zu: %s", i, joint_names[i].c_str());
        }

        auto limits = solver_->getArmJointLimits(arm_side_);
        RCLCPP_INFO(this->get_logger(), "关节限制（从URDF读取）:");
        for (size_t i = 0; i < joint_names.size(); ++i) {
            RCLCPP_INFO(this->get_logger(), "  %s: [%.3f, %.3f]",
                joint_names[i].c_str(), limits.first[i], limits.second[i]);
        }
    }

    void runValidation()
    {
        RCLCPP_INFO(this->get_logger(), "========== FA%s正逆解验证开始 (%s) ==========",
            arm_name_.c_str(), reference_frame_name_.c_str());

        std::string log_filename = std::string("fa_") + (arm_side_ == ArmSide::LEFT ? "left" : "right") + "_arm_ik_failed_cases.log";
        std::ofstream log_file(log_filename, std::ios::out);
        if (log_file.is_open()) {
            log_file << "Reference_Frame, Test_ID, Target_X, Target_Y, Target_Z, Target_R, Target_P, Target_Y, "
                     << "True_q1, True_q2, True_q3, True_q4, True_q5, True_q6, True_q7\n";
        }

        std::random_device rd;
        std::mt19937 gen(rd());

        int success_count = 0;
        double total_time_ms = 0.0;
        long total_steps = 0;

        auto limits = solver_->getArmJointLimits(arm_side_);
        const auto& joint_names = solver_->getArmJointNames(arm_side_);
        for (int test = 0; test < num_tests_; ++test) {
            RCLCPP_INFO(this->get_logger(), "\n--- 测试 %d/%d ---", test + 1, num_tests_);

            Eigen::VectorXd q_rand = sampleRandomJointValues(limits, joint_names, gen);
            logJointVector("随机目标关节角:", joint_names, q_rand);

            auto fk_result = solver_->computeArmFK_SE3(q_rand, arm_side_, kinematics_options_);
            RCLCPP_INFO(this->get_logger(), "FK结果(%s): position=[%.6f, %.6f, %.6f]",
                reference_frame_name_.c_str(), fk_result.p.x(), fk_result.p.y(), fk_result.p.z());

            pinocchio::SE3 T_target = makeTargetPose(fk_result);

            auto start = std::chrono::high_resolution_clock::now();
            IKResult ik_result = solver_->solveArmIK(
                T_target, arm_side_, Eigen::VectorXd(), kinematics_options_);
            auto end = std::chrono::high_resolution_clock::now();

            bool is_failed = false;
            if (!ik_result.has_solution) {
                RCLCPP_ERROR(this->get_logger(), "逆运动学求解失败！");
                is_failed = true;
            } else {
                const Eigen::VectorXd& q_solved = ik_result.q_solution;
                logJointVector("IK求解结果:", joint_names, q_solved);

                auto fk_verify_se3 = solver_->computeArmFK_SE3(q_solved, arm_side_, kinematics_options_);

                double pos_error = (fk_result.p - fk_verify_se3.p).norm();
                Eigen::Matrix3d R_diff = fk_result.R.transpose() * fk_verify_se3.R;
                Eigen::AngleAxisd angle_diff(R_diff);
                double rot_error = std::abs(angle_diff.angle());

                RCLCPP_INFO(this->get_logger(), "一致性验证: pos_error=%.8f m, rot_error=%.8f rad",
                    pos_error, rot_error);

                if (ik_result.success) {
                    std::chrono::duration<double, std::milli> elapsed = end - start;
                    success_count++;
                    total_time_ms += elapsed.count();
                    total_steps += ik_result.iterations;
                    RCLCPP_INFO(this->get_logger(), "  ✅ 正逆解一致，耗时 %.4f ms，迭代 %d 步",
                        elapsed.count(), ik_result.iterations);
                } else {
                    RCLCPP_INFO(this->get_logger(), "  ❌ 未达到精确阈值，返回近似解！");
                    is_failed = true;
                }
            }

            if (is_failed) {
                RCLCPP_WARN(this->get_logger(), "测试 %d 失败，已记录至日志", test + 1);
                if (log_file.is_open()) {
                    log_file << reference_frame_name_ << ", ";
                    writeFailureCase(log_file, test, T_target, q_rand);
                }
            }

            if ((test + 1) % 1000 == 0) {
                RCLCPP_INFO(this->get_logger(), "进度: %d/%d, 当前成功率: %.1f%%",
                    test + 1, num_tests_, 100.0 * success_count / (test + 1));
            }
        }

        RCLCPP_INFO(this->get_logger(), "\n========== 验证结果 ==========");
        RCLCPP_INFO(this->get_logger(), "总测试次数: %d", num_tests_);
        RCLCPP_INFO(this->get_logger(), "成功次数: %d", success_count);
        RCLCPP_INFO(this->get_logger(), "算法分布统计:");
        RCLCPP_INFO(this->get_logger(), "  - 第一阶段 (LDLT) 成功: %d", solver_->stats_.qr_stage_success);
        RCLCPP_INFO(this->get_logger(), "  - 第二阶段 (SVD) 成功: %d", solver_->stats_.svd_stage_success);
        if (success_count > 0) {
            RCLCPP_INFO(this->get_logger(), "平均成功耗时: %.4f ms", total_time_ms / success_count);
            RCLCPP_INFO(this->get_logger(), "平均成功步长: %.2f steps", static_cast<double>(total_steps) / success_count);
            RCLCPP_INFO(this->get_logger(), "成功率: %.1f%%", 100.0 * success_count / num_tests_);
        } else {
            RCLCPP_WARN(this->get_logger(), "没有成功的测试样本，无法统计平均值。");
        }

        if (log_file.is_open()) {
            log_file.close();
            RCLCPP_INFO(this->get_logger(), "失败案例已保存至: %s", log_filename.c_str());
        }
    }

    std::unique_ptr<IKSolver> solver_;
    int num_tests_;
    int max_iters_;
    double eps_;
    ArmSide arm_side_;
    ArmKinematicsOptions kinematics_options_;
    std::string arm_name_;
    std::string reference_frame_name_;
};

int main(int argc, char* argv[])
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<FaArmKinematicNode>());
    rclcpp::shutdown();
    return 0;
}
