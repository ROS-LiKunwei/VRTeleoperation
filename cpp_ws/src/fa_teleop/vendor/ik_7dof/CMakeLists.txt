cmake_minimum_required(VERSION 3.8)
project(ik_7dof)

# 设置 C++ 标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# 1. 查找依赖
find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)
find_package(Python3 REQUIRED COMPONENTS Interpreter Development.Module)
find_package(pybind11 REQUIRED)
find_package(rclcpp REQUIRED)
find_package(Eigen3 REQUIRED)

# ROS Humble's eigenpy exports Boost.Python as python310, which CMake 3.22's
# stock FindBoost cannot map back to boost/python.hpp and reports a warning.
# Prefer eigenpy's patched FindBoost module when it is available.
set(EIGENPY_BOOST_CMAKE_DIR "/opt/ros/humble/lib/x86_64-linux-gnu/cmake/eigenpy/boost")
if(EXISTS "${EIGENPY_BOOST_CMAKE_DIR}/FindBoost.cmake")
  list(PREPEND CMAKE_MODULE_PATH "${EIGENPY_BOOST_CMAKE_DIR}")
endif()

find_package(pinocchio REQUIRED)

find_package(tf2_ros REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(std_srvs REQUIRED)
find_package(srdfdom REQUIRED)

find_package(moveit_ros_planning_interface REQUIRED)
find_package(moveit_core REQUIRED)
find_package(moveit_msgs REQUIRED)
# 2. 创建库 (IK 解算器)

add_library(fa_ik_solver_lib
  src/fa_ik_solver.cpp
)

# 使用 ament 的方式处理依赖，这会自动处理头文件和库

ament_target_dependencies(fa_ik_solver_lib
  Eigen3
  pinocchio
)


target_include_directories(fa_ik_solver_lib
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
    ${EIGEN3_INCLUDE_DIRS}
)

# Python extension for direct access from BeaVR
pybind11_add_module(ik_7dof_pybind
  src/fa_ik_pybind.cpp
)

target_link_libraries(ik_7dof_pybind
  PRIVATE
    fa_ik_solver_lib
)

target_include_directories(ik_7dof_pybind
  PRIVATE
    ${EIGEN3_INCLUDE_DIRS}
)

ament_get_python_install_dir(PYTHON_INSTALL_DIR)
_ament_cmake_python_register_environment_hook()

# 3. 创建可执行文件 (ROS 2 节点)
add_executable(fa_arm_kinematic_node
  src/fa_arm_kinematic_node.cpp
)
add_executable(arm_ik_rviz_node
  src/arm_ik_rviz_node.cpp
)

# 让节点链接到库并获取所有 ROS 2 依赖环境

ament_target_dependencies(fa_arm_kinematic_node
  rclcpp
  Eigen3
  pinocchio
)

ament_target_dependencies(arm_ik_rviz_node
  rclcpp
  Eigen3
  pinocchio
  tf2_ros
  geometry_msgs
  sensor_msgs
  std_srvs
  moveit_ros_planning_interface
  moveit_core
  srdfdom
)

# 链接你自己的库
target_link_libraries(fa_arm_kinematic_node
  fa_ik_solver_lib
)
target_link_libraries(arm_ik_rviz_node
  fa_ik_solver_lib
)

# 4. 安装
install(
  TARGETS
    fa_ik_solver_lib
    fa_arm_kinematic_node
    arm_ik_rviz_node
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION lib/${PROJECT_NAME}
)

install(
  DIRECTORY include/
  DESTINATION include
)

install(
  TARGETS ik_7dof_pybind
  LIBRARY DESTINATION ${PYTHON_INSTALL_DIR}
)

install(
  DIRECTORY  launch 
  DESTINATION share/${PROJECT_NAME}
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # ament_lint_auto_find_test_dependencies()

  add_executable(fa_ik_approx_verify
    src/fa_ik_approx_verify.cpp
  )
  ament_target_dependencies(fa_ik_approx_verify
    Eigen3
    pinocchio
  )
  target_link_libraries(fa_ik_approx_verify
    fa_ik_solver_lib
  )
  target_compile_definitions(fa_ik_approx_verify
    PRIVATE
      IK_7DOF_WORKSPACE_SRC_DIR="${CMAKE_CURRENT_SOURCE_DIR}/.."
  )
  add_test(NAME fa_ik_approx_verify COMMAND fa_ik_approx_verify)
endif()

ament_package()
