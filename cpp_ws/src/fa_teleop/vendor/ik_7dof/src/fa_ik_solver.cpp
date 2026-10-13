#include "ik_7dof/fa_ik_solver.hpp"
#include <pinocchio/parsers/urdf.hpp>
#include <pinocchio/parsers/srdf.hpp>
#include <pinocchio/algorithm/kinematics.hpp>
#include <pinocchio/algorithm/jacobian.hpp>
#include <pinocchio/algorithm/frames.hpp>
#include <pinocchio/algorithm/kinematics-derivatives.hpp>
#include <pinocchio/spatial/log.hpp>
#include <Eigen/Dense>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <memory>
#include <limits>

namespace fa_arm_kinematic
{

// PIMPL实现
// 这个内部类封装了所有Pinocchio相关的实现细节
class IKSolver::Impl
{
public:
    Impl(const std::string& urdf_file,
         const std::string& srdf_file,
         const std::string& pelvis_frame,
         const std::string& left_arm_base_frame,
         const std::string& right_arm_base_frame,
         const std::vector<std::string>& waist_joint_names)
        : urdf_file_(urdf_file),
          srdf_file_(srdf_file),
          pelvis_frame_(pelvis_frame),
          left_arm_base_frame_(left_arm_base_frame),
          right_arm_base_frame_(right_arm_base_frame),
          waist_joint_names_(waist_joint_names)
    {
        // 加载URDF模型
        // buildModel()解析URDF文件，构建运动学树
        pinocchio::urdf::buildModel(urdf_file_, model_);
        
        // 如果提供了SRDF，加载关节限制和默认配置
        if (!srdf_file_.empty()) {
            pinocchio::srdf::loadReferenceConfigurations(model_, srdf_file_, true);
        }
        
        // 3. 创建数据对象
        // Data存储计算过程中的中间结果
        data_ = std::make_unique<pinocchio::Data>(model_);

        pelvis_fid_ = getFrameId(pelvis_frame_);
        left_arm_base_fid_ = getFrameId(left_arm_base_frame_);
        right_arm_base_fid_ = getFrameId(right_arm_base_frame_);
        pelvis_M_left_arm_base_ = computePelvisToFrame(left_arm_base_fid_);
        pelvis_M_right_arm_base_ = computePelvisToFrame(right_arm_base_fid_);
        
        std::cout << "模型加载成功!" << std::endl;
        std::cout << "  关节数量: " << model_.nq << std::endl;
        std::cout << "  框架数量: " << model_.nframes << std::endl;
    }
    
    ~Impl() = default;
    
    // RPY到SE3齐次变换矩阵转换 顺序：Z(roll) * Y(pitch) * X(yaw)
    pinocchio::SE3 rpyzyxToSE3(double x, double y, double z, double roll, double pitch, double yaw)
    {
        Eigen::AngleAxisd Rz(yaw, Eigen::Vector3d::UnitZ());
        Eigen::AngleAxisd Ry(pitch, Eigen::Vector3d::UnitY());
        Eigen::AngleAxisd Rx(roll, Eigen::Vector3d::UnitX());
        Eigen::Matrix3d R = Rz.toRotationMatrix() * Ry.toRotationMatrix() * Rx.toRotationMatrix();
        return pinocchio::SE3(R, Eigen::Vector3d(x, y, z));
    }
    
    // 获取末端执行器Frame ID
    pinocchio::FrameIndex getEeFrameId(const std::string& ee_frame)
    {
        if (model_.existFrame(ee_frame)) {
            return model_.getFrameId(ee_frame);
        }
        throw std::runtime_error("Frame not found: " + ee_frame);
    }

    pinocchio::FrameIndex getFrameId(const std::string& frame) const
    {
        if (model_.existFrame(frame)) {
            return model_.getFrameId(frame);
        }
        throw std::runtime_error("Frame not found: " + frame);
    }

    void setJointsToZero(Eigen::VectorXd& q_full, const std::vector<std::string>& joint_names) const
    {
        for (const auto& jname : joint_names) {
            if (!model_.existJointName(jname)) {
                continue;
            }
            auto jid = model_.getJointId(jname);
            int q_idx = model_.joints[jid].idx_q();
            q_full[q_idx] = 0.0;
        }
    }

    void setWaistJointsToZero(Eigen::VectorXd& q_full) const
    {
        setJointsToZero(q_full, waist_joint_names_);
    }

    pinocchio::SE3 computePelvisToFrame(pinocchio::FrameIndex fid)
    {
        Eigen::VectorXd q_full = pinocchio::neutral(model_);
        setWaistJointsToZero(q_full);
        pinocchio::forwardKinematics(model_, *data_, q_full);
        pinocchio::updateFramePlacements(model_, *data_);

        const pinocchio::SE3 oMp = data_->oMf[pelvis_fid_];
        const pinocchio::SE3 oMf = data_->oMf[fid];
        return oMp.inverse() * oMf;
    }

    pinocchio::SE3 getPelvisToArmBase(ArmSide arm_side) const
    {
        return arm_side == ArmSide::LEFT ? pelvis_M_left_arm_base_ : pelvis_M_right_arm_base_;
    }

    pinocchio::FrameIndex pelvisFrameId() const { return pelvis_fid_; }
    
    // 获取关节索引：建立**“关节名称”与“数值向量索引”**之间的映射
    // 在机器人学中，不同的关节可能有不同的自由度（DoF）。例如，旋转关节占 1 个位置，而球关节可能占 4 个位置（四元数表示 q）和 3 个速度（角速度 v）。因此，不能简单地通过关节 ID 来遍历，必须查找它们在全局向量中的具体索引。
    // q_indices: 配置空间索引(model.nq) 描述机器人姿态，包括关节角/位置以及浮动基姿态（如果有）
    // v_indices: 速度空间索引(model.nv) 描述机器人关节速度，包括关节角速度和基座速度
    bool getJointIndices(const std::vector<std::string>& joint_names,
                     std::vector<int>& q_indices,
                     std::vector<int>& v_indices) const
    {
        q_indices.clear();
        v_indices.clear();
        
        // std::cout << "\n关节索引映射详情:" << std::endl;
        // std::cout << std::left << std::setw(5) << "序号" 
        //         << std::setw(25) << "关节名称" 
        //         << std::setw(10) << "JointID"
        //         << std::setw(10) << "q_idx"
        //         << std::setw(10) << "v_idx"
        //         << std::setw(15) << "下限"
        //         << std::setw(15) << "上限" << std::endl;
        // std::cout << std::string(85, '-') << std::endl;
        
        // int index = 0;
        for (const auto& name : joint_names) {
            if (!model_.existJointName(name)) {
                std::cerr << "警告: 关节 '" << name << "' 未找到!" << std::endl;
                continue;
            }
            pinocchio::JointIndex jid = model_.getJointId(name);
            int q_idx = model_.joints[jid].idx_q();
            int v_idx = model_.joints[jid].idx_v();
            
            q_indices.push_back(q_idx);
            v_indices.push_back(v_idx);
            
            // std::cout << std::left << std::setw(5) << index++
            //         << std::setw(25) << name
            //         << std::setw(10) << jid
            //         << std::setw(10) << q_idx
            //         << std::setw(10) << v_idx
            //         << std::setw(15) << model_.lowerPositionLimit[q_idx]
            //         << std::setw(15) << model_.upperPositionLimit[q_idx]
            //         << std::endl;
        }
        
        return !q_indices.empty();
    }
    
    pinocchio::Model model_;
    std::unique_ptr<pinocchio::Data> data_;
    std::string urdf_file_;
    std::string srdf_file_;
    std::string pelvis_frame_;
    std::string left_arm_base_frame_;
    std::string right_arm_base_frame_;
    std::vector<std::string> waist_joint_names_;

    pinocchio::FrameIndex pelvis_fid_{0};
    pinocchio::FrameIndex left_arm_base_fid_{0};
    pinocchio::FrameIndex right_arm_base_fid_{0};
    pinocchio::SE3 pelvis_M_left_arm_base_;
    pinocchio::SE3 pelvis_M_right_arm_base_;
};

// IKSolver类的公共接口
IKSolver::IKSolver(const std::string& urdf_file, const std::string& srdf_file)
    : urdf_file_(urdf_file), srdf_file_(srdf_file)
{
    comfort_left_q_ref_.resize(7);
    comfort_right_q_ref_.resize(7);
    comfort_left_q_ref_ << -0.5, 0.5, -0.5, -0.5, 0.5, 0.0, 0.0;
    comfort_right_q_ref_ << -0.5, 0.5, 0.5, -0.5, 0.5, 0.0, 0.0;
    pimpl_ = std::make_unique<Impl>(
        urdf_file,
        srdf_file,
        pelvis_frame_,
        left_arm_base_frame_,
        right_arm_base_frame_,
        waist_joints_);
    stats_.reset();
    // 根据实际的 URDF 文件，左臂关节的正确 q_idx 应该是：
    // left_joint1: 需要找到正确的索引
    // shoulder_roll_l_joint: 需要找到正确的索引
    // left_joint3: 需要找到正确的索引
    // ...
    
    // 我们先获取所有关节的索引，然后手动映射
    std::vector<int> temp_q_indices, temp_v_indices;
    pimpl_->getJointIndices(left_arm_joints_, temp_q_indices, temp_v_indices);

    // 启动时打印索引映射，便于验证 Pinocchio 索引是否对应到预期关节
    std::cout << "\n[FA IK] 关节索引映射验证" << std::endl;
    std::cout << std::left
              << std::setw(4)  << "No"
              << std::setw(32) << "JointName"
              << std::setw(8)  << "jid"
              << std::setw(8)  << "q_idx"
              << std::setw(8)  << "v_idx"
              << std::setw(14) << "q_lower"
              << std::setw(14) << "q_upper" << std::endl;
    std::cout << std::string(88, '-') << std::endl;

    for (size_t i = 0; i < left_arm_joints_.size(); ++i) {
        const auto& jname = left_arm_joints_[i];
        if (!pimpl_->model_.existJointName(jname)) {
            std::cout << std::left
                      << std::setw(4)  << i
                      << std::setw(32) << jname
                      << std::setw(8)  << "N/A"
                      << std::setw(8)  << "N/A"
                      << std::setw(8)  << "N/A"
                      << std::setw(14) << "N/A"
                      << std::setw(14) << "N/A" << std::endl;
            continue;
        }

        auto jid = pimpl_->model_.getJointId(jname);
        int q_idx = pimpl_->model_.joints[jid].idx_q();
        int v_idx = pimpl_->model_.joints[jid].idx_v();

        std::cout << std::left
                  << std::setw(4)  << i
                  << std::setw(32) << jname
                  << std::setw(8)  << jid
                  << std::setw(8)  << q_idx
                  << std::setw(8)  << v_idx
                  << std::setw(14) << pimpl_->model_.lowerPositionLimit[q_idx]
                  << std::setw(14) << pimpl_->model_.upperPositionLimit[q_idx]
                  << std::endl;
    }

    if (temp_q_indices.size() != left_arm_joints_.size() ||
        temp_v_indices.size() != left_arm_joints_.size()) {
        throw std::runtime_error(
            "[FA IK] 关节索引数量不匹配，请检查 fa_ik_solver.hpp 里的关节名与 URDF 是否一致");
    }
    
    // 使用获取到的索引
    q_indices_ = temp_q_indices;
    v_indices_ = temp_v_indices;
}

IKSolver::~IKSolver() = default;

void IKSolver::setComfortNullspace(
    double weight,
    const Eigen::VectorXd& left_q_ref,
    const Eigen::VectorXd& right_q_ref)
{
    if (left_q_ref.size() != 7 || right_q_ref.size() != 7
        || !left_q_ref.allFinite() || !right_q_ref.allFinite()) {
        throw std::invalid_argument("FA IK comfort references must contain 7 finite values per arm");
    }
    comfort_nullspace_weight_ = std::max(0.0, weight);
    comfort_left_q_ref_ = left_q_ref;
    comfort_right_q_ref_ = right_q_ref;
}

Eigen::VectorXd IKSolver::solveIK_Core(
    const pinocchio::SE3& T_target,
    const Eigen::VectorXd& q_init,
    int max_iters,
    double eps,
    int& iters_out,
    SolverMethod method,
    ArmSide arm_side,
    const ArmKinematicsOptions& options,
    bool* exact_solution,
    bool* acceptable_solution,
    double* best_error)
{
    using namespace pinocchio;
    using namespace Eigen;

    Model& model = pimpl_->model_;
    Data& data = *pimpl_->data_;
    const std::string& ee_frame = arm_side == ArmSide::LEFT ? left_ee_frame_ : right_ee_frame_;
    const std::vector<std::string>& arm_joints = arm_side == ArmSide::LEFT ? left_arm_joints_ : right_arm_joints_;
    FrameIndex ee_fid = pimpl_->getEeFrameId(ee_frame);
    FrameIndex pelvis_fid = pimpl_->pelvisFrameId();
    const SE3 pelvis_M_arm_base = pimpl_->getPelvisToArmBase(arm_side);
    const int n_arm = arm_joints.size();

    VectorXd q = neutral(model);
    pimpl_->setWaistJointsToZero(q);
    for (int i = 0; i < n_arm; ++i) {
        int qi = model.joints[model.getJointId(arm_joints[i])].idx_q();
        q[qi] = q_init[i];
    }

    // --- 根据模式设置初始阻 ---
    // 快速模式 (LDLT/QR) 建议阻尼稍微大一点，增加稳定性
    // 稳健模式 (SVD) 阻尼可以小一点，因为 SVD 本身有截断逻辑
    double lambda = method == SolverMethod::SVD ? 1e-3 : 1e-2;
    VectorXd grad_H = VectorXd::Zero(n_arm);
    VectorXd best_q_arm = VectorXd::Zero(n_arm);
    double best_error_norm = std::numeric_limits<double>::infinity();
    bool best_acceptable_solution = false;
    const double convergence_eps = std::max(eps, std::numeric_limits<double>::epsilon());

    if (exact_solution) {
        *exact_solution = false;
    }
    if (acceptable_solution) {
        *acceptable_solution = false;
    }

    for (int iter = 0; iter < max_iters; ++iter) {
        if (std::chrono::steady_clock::now() >= options.deadline) {
            break;
        }
        iters_out = iter + 1;

        forwardKinematics(model, data, q);
        updateFramePlacements(model, data);

        const SE3 oMp = data.oMf[pelvis_fid];
        SE3 oT_target;
        if (options.reference_frame == ArmReferenceFrame::ARM_BASE) {
            const SE3 oMarm_base = oMp * pelvis_M_arm_base;
            oT_target = oMarm_base * T_target;
        } else {
            oT_target = oMp * T_target;
        }

        SE3 T_current = data.oMf[ee_fid];
        Motion err_motion = log6(T_current.actInv(oT_target));
        VectorXd err = err_motion.toVector(); // 6维向量: [v_x, v_y, v_z, w_x, w_y, w_z]

        double error_norm = err.norm();
        const double position_weight = std::max(0.0, options.position_weight);
        const double orientation_weight = std::max(0.0, options.orientation_weight);
        Eigen::VectorXd weights(6);
        if (error_norm > 0.01) {
            // 第一阶段：粗调。位置保持完整权重，姿态先弱约束。
            // 优先把末端拉到目标点附近，避免姿态奇异点把冗余臂带到伸直/坏分支。
            weights << position_weight, position_weight, position_weight,
                0.4 * orientation_weight, 0.4 * orientation_weight, 0.4 * orientation_weight;
        } else {
            // 第二阶段：精调。位置已接近 1cm 内后恢复姿态权重。
            // 精细操作中姿态和位置同样重要，最终收敛不能长期弱化姿态项。
            weights << position_weight, position_weight, position_weight,
                orientation_weight, orientation_weight, orientation_weight;
        }
        double weighted_error_norm = err.cwiseProduct(weights).norm();
        double position_error = err.head<3>().norm();
        double orientation_error = err.tail<3>().norm();

        // eps 是未加权 SE(3) 末端误差范数的收敛要求。
        const bool exact_reached = error_norm <= convergence_eps;
        const bool acceptable_reached =
            position_error <= std::max(0.0, options.acceptable_position_error)
            && orientation_error <= std::max(0.0, options.acceptable_orientation_error);

        // 记录当前迭代中误差最小的关节角，目标不可达时返回这个最优近似解。
        if (weighted_error_norm < best_error_norm) {
            best_error_norm = weighted_error_norm;
            best_acceptable_solution = acceptable_reached;
            for (int i = 0; i < n_arm; ++i) {
                best_q_arm[i] = q[model.joints[model.getJointId(arm_joints[i])].idx_q()];
            }
        }

        // acceptable_* 只标记最大迭代后的最佳近似解，不再提前终止精调。
        if (exact_reached) {
            VectorXd res(n_arm);
            for (int i = 0; i < n_arm; ++i)
                res[i] = q[model.joints[model.getJointId(arm_joints[i])].idx_q()];
            if (exact_solution) {
                *exact_solution = true;
            }
            if (acceptable_solution) {
                *acceptable_solution = true;
            }
            if (best_error) {
                *best_error = error_norm;
            }
            return res;
        }

        // 应用权重
        VectorXd weighted_err = err.cwiseProduct(weights); // coefficient-wise product，逐元素对应相乘，并生成一个新的同维度向量

        // 计算雅可比
        MatrixXd J_full = MatrixXd::Zero(6, model.nv);
        computeFrameJacobian(model, data, q, ee_fid, LOCAL, J_full);
        
        MatrixXd J_arm(6, n_arm);
        for (int i = 0; i < n_arm; ++i) {
            J_arm.col(i) = J_full.col(model.joints[model.getJointId(arm_joints[i])].idx_v());
        }

        // =====================================================================
        // 统一计算零空间避障斥力 (grad_H) && 舒适姿态拉力
        // =====================================================================
        grad_H.setZero(); 
        double repulsion_gain = 0.0;
        
        // 奇异感知：计算雅可比矩阵的最小奇异值，在奇异位形时减小零空间拉力
        // 避免零空间坍缩时舒适拉力泄漏到任务空间导致抖动
        double singularity_scale = 1.0;
        {
            Eigen::JacobiSVD<Eigen::MatrixXd> svd(J_arm);
            Eigen::VectorXd sv = svd.singularValues();
            double min_sv = sv.minCoeff();
            double singularity_threshold = 0.05;
            singularity_scale = std::min(1.0, min_sv / singularity_threshold);
        }
        
        // 平滑启停零空间项，避免 error_norm 在阈值附近来回穿越时突然开/关导致抖动。
        // 3cm 内基本关闭，8cm 外完全启用，中间用 smoothstep 连续过渡。
        const double nullspace_ramp_start = 3e-2;
        const double nullspace_ramp_end = 8e-2;
        double nullspace_activation = 0.0;
        if (error_norm >= nullspace_ramp_end) {
            nullspace_activation = 1.0;
        } else if (error_norm > nullspace_ramp_start) {
            const double t = (error_norm - nullspace_ramp_start)
                / (nullspace_ramp_end - nullspace_ramp_start);
            nullspace_activation = t * t * (3.0 - 2.0 * t);
        }

        VectorXd seed_pull = VectorXd::Zero(n_arm);
        if (nullspace_activation > 0.0 || options.continuity_nullspace_weight > 0.0) {
            auto limits = getArmJointLimits(arm_side);
            VectorXd q_arm(n_arm);
            for(int i=0; i<n_arm; ++i){
                q_arm[i] = q[model.joints[model.getJointId(arm_joints[i])].idx_q()];
            } 

            const VectorXd& q_rest = arm_side == ArmSide::RIGHT
                ? comfort_right_q_ref_
                : comfort_left_q_ref_;
            // 配置权重只作用于零空间舒适姿态，不改变末端任务误差权重。
            const double k_rest = comfort_nullspace_weight_;

            const double threshold_percent = 0.15; // 15% 的边缘触发力场
            const double limit_gain = 0.4;

            for(int i=0; i<n_arm; ++i) {
                double dist_to_upper = limits.second[i] - q_arm[i];
                double dist_to_lower = q_arm[i] - limits.first[i];
                double range = limits.second[i] - limits.first[i];
                double margin = range * threshold_percent; 
                
                // 计算限位斥力 (高优)   斥力 = 增益系数 * ((边界厚度 - 当前距离) / 边界厚度)^2
                double limit_repulsion = 0.0;
                if (dist_to_upper < margin) {
                    const double x = (margin - dist_to_upper) / margin;
                    limit_repulsion = -limit_gain * x * x;
                } else if (dist_to_lower < margin) {
                    const double x = (margin - dist_to_lower) / margin;
                    limit_repulsion = limit_gain * x * x;
                }
                
                // 计算饱和舒适姿态拉力 (低优)
                // 使用 tanh 函数代替线性函数，限制最大拉力，避免抖动
                double comfort_pull = -k_rest * std::tanh(2.0 * (q_arm[i] - q_rest[i]));

                // 叠加梯度：当靠近限位时，limit_repulsion 会急剧变大，掩盖住 rest 姿态拉力
                grad_H[i] = nullspace_activation * (limit_repulsion + comfort_pull);
                seed_pull[i] = -std::tanh(2.0 * (q_arm[i] - q_init[i]));
            }

            // 对斥力梯度做简单的限幅
            double max_grad = 10.0;
            for(int i=0; i<n_arm; ++i) 
                grad_H[i] = std::max(-max_grad, std::min(max_grad, grad_H[i]));

            repulsion_gain = singularity_scale * 0.05;
        }

        // 舒适姿态项在奇异区衰减；连续性项以很小的独立增益保留，
        // 防止翻腕经过奇异区时突然切换到远端等价分支。
        VectorXd g0 = repulsion_gain * grad_H
            + options.continuity_nullspace_weight * seed_pull;
        VectorXd dq;
        
        // =====================================================================
        // LDLT 和 SVD 的求解分支
        // =====================================================================
        if (method == SolverMethod::LDLT) {
            MatrixXd JJt = J_arm * J_arm.transpose();
            JJt.diagonal().array() += (lambda * lambda);
            
            // 使用数学等效变形：dq = J# * e + (I - J#*J)*g0 转换为 dq = J# * (e - J*g0) + g0
            // 这样能在一行 LDLT 求解中同时实现：DLS 阻尼 + 零空间避障
            VectorXd modified_err = weighted_err - J_arm * g0;
            dq = J_arm.transpose() * JJt.ldlt().solve(modified_err) + g0;

        } else {
            JacobiSVD<MatrixXd> svd(J_arm, ComputeThinU | ComputeThinV);
            VectorXd sv = svd.singularValues();
            VectorXd sv_inv = sv;
            for (int i = 0; i < sv.size(); ++i) {
                sv_inv(i) = (sv(i) > 1e-4) ? (1.0 / sv(i)) : 0.0;
            }

            MatrixXd J_pinv = svd.matrixV() * sv_inv.asDiagonal() * svd.matrixU().transpose();
            MatrixXd NullSpace = MatrixXd::Identity(n_arm, n_arm) - J_pinv * J_arm;
            
            // SVD 下直接使用计算好的伪逆和零空间矩阵
            dq = J_pinv * weighted_err + NullSpace * g0;
        }

        // --- 自适应步长 ---
        double adaptive_dt = (error_norm > 1e-1) ? 0.2 : 0.6;
        VectorXd delta_q = dq * adaptive_dt;
        // 对齐 sysmo32 实时 IK：每轮迭代限制关节增量，近似解也必须连续。
        const double max_joint_step_rad = 1.0;
        const double delta_norm = delta_q.norm();
        if (delta_norm > max_joint_step_rad) {
            delta_q *= max_joint_step_rad / delta_norm;
        }

        // 更新 q 并截断限位
        for (int i = 0; i < n_arm; ++i) {
            int qi = model.joints[model.getJointId(arm_joints[i])].idx_q();
            q[qi] += delta_q[i];
            const double soft_margin = 0.05;  // rad，避免 IK 结果贴住硬限位
            const double lower = model.lowerPositionLimit[qi] + soft_margin;
            const double upper = model.upperPositionLimit[qi] - soft_margin;
            if (lower < upper) {
                q[qi] = std::max(lower, std::min(upper, q[qi]));
            } else {
                q[qi] = std::max(model.lowerPositionLimit[qi], std::min(model.upperPositionLimit[qi], q[qi]));
            }
        }

        // 只有当误差确实很小时，才减小阻尼以加速最后收敛
        if (error_norm < 1e-2) {
            lambda = method == SolverMethod::SVD ? 1e-4 : 1e-3;
            adaptive_dt = 1.0;
        }
    }

    if (best_error) {
        *best_error = best_error_norm;
    }
    if (acceptable_solution) {
        *acceptable_solution = best_acceptable_solution;
    }

    return std::isfinite(best_error_norm) ? best_q_arm : VectorXd();
}



IKResult IKSolver::solveArmIK(
    const pinocchio::SE3 T_target_pose,
    const ArmSide arm_side,
    const Eigen::VectorXd initial_q,
    const ArmKinematicsOptions options,
    const int max_iters,
    const double eps)
{
    IKResult ik_result;
    int total_iters = 0;
    int current_iters = 0;

    // --- 姿态正交化/四元数化 ---
    // 目的：修正 RPY 转换过程中可能产生的微小非正交误差，确保旋转矩阵严格有效
    pinocchio::SE3 T_target = T_target_pose;
    Eigen::Matrix3d R = T_target.rotation();
    Eigen::Quaterniond q_rot(R);     // 将矩阵转为四元数
    q_rot.normalize();               // 归一化四元数
    T_target.rotation(q_rot.toRotationMatrix()); // 转回旋转矩阵并重新赋值给 T_target

    // 2. 准备初始值
    auto limits = getArmJointLimits(arm_side);
    Eigen::VectorXd q_start = (initial_q.size() == 7) ? initial_q : (limits.first + limits.second) / 2.0;

    // 3. 第一次尝试(尝试快速求解 (LDLT))
    const int per_stage_iter_budget = std::max(max_iters, 1);
    int first_stage_iters = per_stage_iter_budget;
    bool exact_solution = false;
    bool acceptable_solution = false;
    double approx_error = std::numeric_limits<double>::infinity();
    double best_approx_error = std::numeric_limits<double>::infinity();
    Eigen::VectorXd best_approx;
    Eigen::VectorXd result = solveIK_Core(
        T_target, q_start, first_stage_iters, eps, current_iters, SolverMethod::LDLT,
        arm_side, options, &exact_solution, &acceptable_solution, &approx_error);
    total_iters += current_iters;

    if (acceptable_solution && result.size() > 0){
        stats_.qr_stage_success++; // LDLT 阶段成功
        ik_result.success = exact_solution;
        ik_result.has_solution = true;
        ik_result.q_solution = result;
        ik_result.iterations = total_iters;
        PoseSE3 pose = computeArmFK_SE3(result, arm_side, options);
        ik_result.position_error = (T_target.translation() - pose.p).norm();
        Eigen::AngleAxisd angle_diff(T_target.rotation().transpose() * pose.R);
        ik_result.orientation_error = std::abs(angle_diff.angle());
        return ik_result;
    }
    if (result.size() > 0 && approx_error < best_approx_error) {
        best_approx_error = approx_error;
        best_approx = result;
    }

    // 实时遥操作可以显式跳过 SVD fallback，优先返回当前连续初值附近的 LDLT 近似解。
    // SVD 更稳健，但可能跳到另一组等价 IK 分支，表现为手臂突然卡住或大幅跳变。
    if ((options.skip_svd_fallback || per_stage_iter_budget <= 10) && best_approx.size() > 0) {
        stats_.total_failures++;
        ik_result.success = false;
        ik_result.has_solution = true;
        ik_result.q_solution = best_approx;
        ik_result.iterations = total_iters;
        PoseSE3 pose = computeArmFK_SE3(best_approx, arm_side, options);
        ik_result.position_error = (T_target.translation() - pose.p).norm();
        Eigen::AngleAxisd angle_diff(T_target.rotation().transpose() * pose.R);
        ik_result.orientation_error = std::abs(angle_diff.angle());
        return ik_result;
    }

    // --- 第二阶段：快速求解失败，只用当前连续初值做一次 SVD 稳健求解 ---
    // 在线遥操作需要时间连续性。随机重启虽然能提高离线精确解概率，
    // 但会显著增加耗时，并且可能在等价 IK 分支之间跳变。
    Eigen::VectorXd svd_start = best_approx.size() == 7 ? best_approx : q_start;
    result = solveIK_Core(
        T_target, svd_start, per_stage_iter_budget, eps, current_iters, SolverMethod::SVD,
        arm_side, options, &exact_solution, &acceptable_solution, &approx_error);
    total_iters += current_iters;

    if (acceptable_solution && result.size() > 0) {
        stats_.svd_stage_success++;
        ik_result.success = exact_solution;
        ik_result.has_solution = true;
        ik_result.q_solution = result;
        ik_result.iterations = total_iters;
        PoseSE3 pose = computeArmFK_SE3(result, arm_side, options);
        ik_result.position_error = (T_target.translation() - pose.p).norm();
        Eigen::AngleAxisd angle_diff(T_target.rotation().transpose() * pose.R);
        ik_result.orientation_error = std::abs(angle_diff.angle());
        return ik_result;
    }
    if (result.size() > 0 && approx_error < best_approx_error) {
        best_approx_error = approx_error;
        best_approx = result;
    }

    stats_.total_failures++;
    ik_result.success = false;
    ik_result.has_solution = best_approx.size() > 0;
    ik_result.q_solution = best_approx;
    ik_result.iterations = total_iters;
    if (ik_result.has_solution) {
        PoseSE3 pose = computeArmFK_SE3(best_approx, arm_side, options);
        ik_result.position_error = (T_target.translation() - pose.p).norm();
        Eigen::AngleAxisd angle_diff(T_target.rotation().transpose() * pose.R);
        ik_result.orientation_error = std::abs(angle_diff.angle());
    }
    return ik_result; // 目标不可达时返回所有尝试中的最佳近似解
}

PoseRPY IKSolver::computeArmFK(const Eigen::VectorXd& q, ArmSide arm_side,
                               const ArmKinematicsOptions& options)
{
    PoseSE3 pose_se3 = computeArmFK_SE3(q, arm_side, options);
    PoseRPY pose;
    Eigen::Vector3d t = pose_se3.p;
    pose.x = t.x();
    pose.y = t.y();
    pose.z = t.z();
    
    Eigen::Matrix3d R = pose_se3.R;
    if (std::abs(R(2,0)) < 1.0 - 1e-6) {
        pose.pitch = std::asin(-R(2,0));
        pose.roll  = std::atan2(R(2,1), R(2,2));
        pose.yaw   = std::atan2(R(1,0), R(0,0));
    } else {
        pose.pitch = (R(2,0) <= -1.0) ? M_PI/2 : -M_PI/2;
        pose.roll  = 0.0;
        pose.yaw   = std::atan2(-R(0,1), R(1,1));
    }
    
    return pose;
}

std::pair<Eigen::VectorXd, Eigen::VectorXd> IKSolver::getArmJointLimits(ArmSide arm_side) const {
    const std::vector<std::string>& arm_joints = arm_side == ArmSide::LEFT ? left_arm_joints_ : right_arm_joints_;
    size_t n = arm_joints.size();
    Eigen::VectorXd lower(n), upper(n);
    
    for (size_t i = 0; i < n; ++i) {
        // 关键：通过关节名称获取模型中的真实 ID
        if (!pimpl_->model_.existJointName(arm_joints[i])) {
            continue; 
        }
        auto jid = pimpl_->model_.getJointId(arm_joints[i]);
        // 获取该关节在全局 q 向量中的起始索引
        int q_idx = pimpl_->model_.joints[jid].idx_q();
        
        // 直接从 Pinocchio 加载的 URDF 数据中读取
        lower[i] = pimpl_->model_.lowerPositionLimit[q_idx];
        upper[i] = pimpl_->model_.upperPositionLimit[q_idx];
    }
    return {lower, upper};
}

PoseSE3 IKSolver::computeArmFK_SE3(const Eigen::VectorXd& q, ArmSide arm_side,
                                   const ArmKinematicsOptions& options)
{
    using namespace pinocchio;
    
    // 1. 将 7 维 q 映射到全量 q_full (处理有腰部或其它关节的情况)
    Eigen::VectorXd q_full = neutral(pimpl_->model_);
    pimpl_->setWaistJointsToZero(q_full);
    const std::vector<std::string>& arm_joints = arm_side == ArmSide::LEFT ? left_arm_joints_ : right_arm_joints_;
    for (size_t i = 0; i < arm_joints.size(); ++i) {
        int qi = pimpl_->model_.joints[pimpl_->model_.getJointId(arm_joints[i])].idx_q();
        q_full[qi] = q[i];
    }

    // 2. 更新运动学
    forwardKinematics(pimpl_->model_, *pimpl_->data_, q_full);
    updateFramePlacements(pimpl_->model_, *pimpl_->data_);
    
    // 3. 获取末端 Frame 的 SE3 变换
    const std::string& ee_frame = arm_side == ArmSide::LEFT ? left_ee_frame_ : right_ee_frame_;
    FrameIndex ee_fid = pimpl_->getEeFrameId(ee_frame);
    FrameIndex pelvis_fid = pimpl_->pelvisFrameId();
    const SE3 oMp = pimpl_->data_->oMf[pelvis_fid];
    const SE3 oT = pimpl_->data_->oMf[ee_fid];

    SE3 T;
    if (options.reference_frame == ArmReferenceFrame::ARM_BASE) {
        const SE3 pelvis_M_arm_base = pimpl_->getPelvisToArmBase(arm_side);
        const SE3 oMarm_base = oMp * pelvis_M_arm_base;
        T = oMarm_base.inverse() * oT;
    } else {
        T = oMp.inverse() * oT;
    }
    
    // 4. 封装结果
    PoseSE3 pose;
    pose.p = T.translation(); // 提取位置
    pose.R = T.rotation();    // 提取旋转矩阵
    
    return pose;
}

const std::vector<std::string>& IKSolver::getArmJointNames(ArmSide arm_side) const {
    return arm_side == ArmSide::LEFT ? left_arm_joints_ : right_arm_joints_;
}

size_t IKSolver::getArmJointCount(ArmSide arm_side) const {
    return arm_side == ArmSide::LEFT ? left_arm_joints_.size() : right_arm_joints_.size();
}

} // namespace fa_arm_kinematic
