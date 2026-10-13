cmake_minimum_required(VERSION 3.8)
project(min_snap)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(std_srvs REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/MinSnapTarget.msg"
)

add_library(min_snap_core
  src/fa_joint_mapping.cpp
  src/min_snap_trajectory.cpp
  src/min_snap_planner.cpp
  src/trajectory_recorder.cpp
)
target_compile_features(min_snap_core PUBLIC cxx_std_17)
target_include_directories(min_snap_core PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)
ament_target_dependencies(min_snap_core
  rclcpp
  sensor_msgs
  std_msgs
  std_srvs
)

add_executable(min_snap_node src/min_snap_node.cpp)
target_compile_features(min_snap_node PUBLIC cxx_std_17)
target_include_directories(min_snap_node PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)
ament_target_dependencies(min_snap_node
  rclcpp
  sensor_msgs
  std_msgs
  std_srvs
)
rosidl_get_typesupport_target(cpp_typesupport_target ${PROJECT_NAME} "rosidl_typesupport_cpp")
target_link_libraries(min_snap_node min_snap_core "${cpp_typesupport_target}")

add_executable(pause_publish_client src/pause_publish_client.cpp)
target_compile_features(pause_publish_client PUBLIC cxx_std_17)
ament_target_dependencies(pause_publish_client
  rclcpp
  std_srvs
)

add_executable(resume_publish_client src/resume_publish_client.cpp)
target_compile_features(resume_publish_client PUBLIC cxx_std_17)
ament_target_dependencies(resume_publish_client
  rclcpp
  std_srvs
)

install(TARGETS
  min_snap_core
  min_snap_node
  pause_publish_client
  resume_publish_client
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY include/
  DESTINATION include
)

install(DIRECTORY config
  DESTINATION share/${PROJECT_NAME}
)

install(DIRECTORY launch
  DESTINATION share/${PROJECT_NAME}
)

install(PROGRAMS
  scripts/benchmark_min_snap_cpu.py
  scripts/monitor_min_snap_cpu.py
  scripts/plot_min_snap_tracking.py
  scripts/test_min_snap_online_scenarios.py
  DESTINATION lib/${PROJECT_NAME}
)

ament_export_include_directories(include)
ament_export_libraries(min_snap_core)
ament_export_dependencies(rclcpp sensor_msgs std_msgs std_srvs rosidl_default_runtime)
ament_package()
