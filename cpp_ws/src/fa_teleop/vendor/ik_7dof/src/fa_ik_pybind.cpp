#include <chrono>
#include <cmath>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include <Eigen/Dense>
#include <pinocchio/spatial/se3.hpp>
#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "ik_7dof/fa_ik_solver.hpp"

namespace py = pybind11;
using namespace fa_arm_kinematic;

namespace
{
ArmSide parseArmSide(const std::string& side)
{
    if (side == "left") {
        return ArmSide::LEFT;
    }
    if (side == "right") {
        return ArmSide::RIGHT;
    }
    throw std::invalid_argument("arm_side must be 'left' or 'right'");
}

ArmKinematicsOptions parseOptions(
    const std::string& reference_frame,
    bool skip_svd_fallback = false,
    double position_weight = 1.0,
    double orientation_weight = 1.0,
    double acceptable_position_error = 0.05,
    double acceptable_orientation_error = 0.05)
{
    ArmKinematicsOptions options;
    if (reference_frame == "pelvis") {
        options.reference_frame = ArmReferenceFrame::PELVIS;
    } else if (reference_frame == "arm_base") {
        options.reference_frame = ArmReferenceFrame::ARM_BASE;
    } else {
        throw std::invalid_argument("reference_frame must be 'pelvis' or 'arm_base'");
    }
    options.skip_svd_fallback = skip_svd_fallback;
    options.position_weight = position_weight;
    options.orientation_weight = orientation_weight;
    options.acceptable_position_error = acceptable_position_error;
    options.acceptable_orientation_error = acceptable_orientation_error;
    return options;
}

Eigen::VectorXd vectorFromStd(const std::vector<double>& values, const char* label, bool allow_empty = false)
{
    if (allow_empty && values.empty()) {
        return Eigen::VectorXd();
    }
    if (values.size() != 7) {
        throw std::invalid_argument(std::string(label) + " must contain 7 values");
    }
    Eigen::VectorXd q(7);
    for (size_t i = 0; i < values.size(); ++i) {
        q[static_cast<int>(i)] = values[i];
    }
    return q;
}

double finiteDoubleFromObject(const py::handle& value, const char* label)
{
    const double parsed = py::cast<double>(value);
    if (!std::isfinite(parsed)) {
        throw std::invalid_argument(std::string(label) + " contains NaN/Inf");
    }
    return parsed;
}

py::sequence sequenceFromObject(const py::object& values, const char* label)
{
    if (!PySequence_Check(values.ptr())) {
        throw std::invalid_argument(std::string(label) + " must be a Python sequence");
    }
    return py::reinterpret_borrow<py::sequence>(values);
}

Eigen::Vector3d vector3FromObject(const py::object& values, const char* label)
{
    py::sequence seq = sequenceFromObject(values, label);
    if (py::len(seq) != 3) {
        throw std::invalid_argument(std::string(label) + " must contain 3 values");
    }

    Eigen::Vector3d out;
    for (py::ssize_t i = 0; i < 3; ++i) {
        out[static_cast<int>(i)] = finiteDoubleFromObject(seq[i], label);
    }
    return out;
}

Eigen::Matrix3d matrix3FromObject(const py::object& values, const char* label)
{
    py::sequence seq = sequenceFromObject(values, label);
    const py::ssize_t outer_len = py::len(seq);

    Eigen::Matrix3d out;
    if (outer_len == 3 && PySequence_Check(seq[0].ptr())) {
        for (py::ssize_t r = 0; r < 3; ++r) {
            py::sequence row = sequenceFromObject(py::reinterpret_borrow<py::object>(seq[r]), label);
            if (py::len(row) != 3) {
                throw std::invalid_argument(std::string(label) + " must be a 3x3 matrix");
            }
            for (py::ssize_t c = 0; c < 3; ++c) {
                out(static_cast<int>(r), static_cast<int>(c)) = finiteDoubleFromObject(row[c], label);
            }
        }
        return out;
    }

    if (outer_len == 9) {
        for (py::ssize_t i = 0; i < 9; ++i) {
            out(static_cast<int>(i / 3), static_cast<int>(i % 3)) = finiteDoubleFromObject(seq[i], label);
        }
        return out;
    }

    throw std::invalid_argument(std::string(label) + " must be a 3x3 matrix or 9-value sequence");
}
}  // namespace

class PyFaIkSolver
{
public:
    PyFaIkSolver(const std::string& urdf_file, const std::string& srdf_file)
        : solver_(std::make_unique<IKSolver>(urdf_file, srdf_file))
    {
    }

    void configureComfortNullspace(
        double weight,
        const std::vector<double>& left_q_ref,
        const std::vector<double>& right_q_ref)
    {
        solver_->setComfortNullspace(
            weight,
            vectorFromStd(left_q_ref, "left_q_ref"),
            vectorFromStd(right_q_ref, "right_q_ref"));
    }

    py::dict solveArmIk(
        const py::object& translation_obj,
        const py::object& rotation_obj,
        const std::string& arm_side,
        const std::vector<double>& initial_q,
        const std::string& reference_frame,
        int max_iters,
        double eps,
        bool skip_svd_fallback,
        double position_weight,
        double orientation_weight,
        double acceptable_position_error,
        double acceptable_orientation_error,
        double continuity_nullspace_weight)
    {
        const Eigen::Vector3d translation = vector3FromObject(translation_obj, "translation");
        const Eigen::Matrix3d rotation = matrix3FromObject(rotation_obj, "rotation");
        pinocchio::SE3 target(rotation, translation);

        const auto side = parseArmSide(arm_side);
        auto options = parseOptions(
            reference_frame,
            skip_svd_fallback,
            position_weight,
            orientation_weight,
            acceptable_position_error,
            acceptable_orientation_error);
        options.continuity_nullspace_weight = std::max(0.0, continuity_nullspace_weight);
        Eigen::VectorXd q_init = vectorFromStd(initial_q, "initial_q", true);

        const auto start = std::chrono::steady_clock::now();
        IKResult result = solver_->solveArmIK(target, side, q_init, options, max_iters, eps);
        const auto end = std::chrono::steady_clock::now();
        const double elapsed_ms = std::chrono::duration<double, std::milli>(end - start).count();

        std::vector<double> q;
        if (result.q_solution.size() == 7) {
            q.reserve(7);
            for (int i = 0; i < result.q_solution.size(); ++i) {
                q.push_back(result.q_solution[i]);
            }
        }

        py::dict out;
        out["success"] = result.success;
        out["has_solution"] = result.has_solution;
        out["q_target"] = q;
        out["position_error"] = result.position_error;
        out["orientation_error"] = result.orientation_error;
        out["iterations"] = result.iterations;
        out["solve_time_ms"] = elapsed_ms;
        return out;
    }

    py::dict computeArmFk(
        const std::vector<double>& q_values,
        const std::string& arm_side,
        const std::string& reference_frame)
    {
        const auto q = vectorFromStd(q_values, "q");
        const auto side = parseArmSide(arm_side);
        const auto options = parseOptions(reference_frame);
        PoseSE3 pose = solver_->computeArmFK_SE3(q, side, options);

        py::dict out;
        out["translation"] = pose.p;
        out["rotation"] = pose.R;
        return out;
    }

private:
    std::unique_ptr<IKSolver> solver_;
};

PYBIND11_MODULE(ik_7dof_pybind, m)
{
    m.doc() = "pybind11 wrapper for ik_7dof/fa_ik_solver.hpp";
    py::class_<PyFaIkSolver>(m, "FaIkSolver")
        .def(py::init<const std::string&, const std::string&>(), py::arg("urdf_file"), py::arg("srdf_file") = "")
        .def("configure_comfort_nullspace", &PyFaIkSolver::configureComfortNullspace,
             py::arg("weight"),
             py::arg("left_q_ref"),
             py::arg("right_q_ref"))
        .def("solve_arm_ik", &PyFaIkSolver::solveArmIk,
             py::arg("translation"),
             py::arg("rotation"),
             py::arg("arm_side"),
             py::arg("initial_q"),
             py::arg("reference_frame") = "pelvis",
             py::arg("max_iters") = 200,
             py::arg("eps") = 1e-3,
             py::arg("skip_svd_fallback") = false,
             py::arg("position_weight") = 1.0,
             py::arg("orientation_weight") = 1.0,
             py::arg("acceptable_position_error") = 0.05,
             py::arg("acceptable_orientation_error") = 0.05,
             py::arg("continuity_nullspace_weight") = 0.0)
        .def("compute_arm_fk", &PyFaIkSolver::computeArmFk,
             py::arg("q"),
             py::arg("arm_side"),
             py::arg("reference_frame") = "pelvis");
}
