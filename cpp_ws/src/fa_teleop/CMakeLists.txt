cmake_minimum_required(VERSION 3.16)
project(fa_teleop VERSION 1.0.0 LANGUAGES C CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(ament_index_cpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(std_srvs REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(Python3 REQUIRED COMPONENTS Interpreter)
# ROS Humble's rosidl generator still uses the legacy FindPythonInterp variable.
set(PYTHON_EXECUTABLE "${Python3_EXECUTABLE}" CACHE FILEPATH "ROS Python interpreter" FORCE)
find_package(pinocchio REQUIRED)
# eigenpy exposes a dot-prefixed PYTHON_SOABI; ROS's FindPythonExtra adds its own dot.
execute_process(COMMAND ${Python3_EXECUTABLE} -c "import sysconfig; print(sysconfig.get_config_var('SOABI'))"
  OUTPUT_VARIABLE PYTHON_SOABI OUTPUT_STRIP_TRAILING_WHITESPACE COMMAND_ERROR_IS_FATAL ANY)
find_package(yaml-cpp REQUIRED)
find_package(OpenSSL REQUIRED)
find_package(Threads REQUIRED)
find_library(ZMQ_LIBRARY zmq REQUIRED)

# Reconfigure whenever any versioned input changes. No generated files in the source tree.
file(GLOB_RECURSE BUNDLE_INPUTS CONFIGURE_DEPENDS
  "include/*" "src/*" "vendor/*" "models/*" "config/*" "msg/*" "tools/*" "tests/*")
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS ${BUNDLE_INPUTS} ORIGINS.json CMakeLists.txt package.xml)
execute_process(COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tools/lock_bundle.py
  --source ${CMAKE_CURRENT_SOURCE_DIR} --output ${CMAKE_CURRENT_BINARY_DIR}/bundle
  RESULT_VARIABLE LOCK_RESULT)
if(NOT LOCK_RESULT EQUAL 0)
  message(FATAL_ERROR "Cannot generate version-locked deployment bundle")
endif()

rosidl_generate_interfaces(${PROJECT_NAME} "msg/JointTarget.msg" DEPENDENCIES std_msgs)
rosidl_get_typesupport_target(typesupport ${PROJECT_NAME} rosidl_typesupport_cpp)

# Static libraries prevent a stale external ik_7dof/min_snap .so from being selected by LD_LIBRARY_PATH.
add_library(fa_ik_locked STATIC vendor/ik_7dof/src/fa_ik_solver.cpp)
target_include_directories(fa_ik_locked PUBLIC vendor/ik_7dof/include)
target_link_libraries(fa_ik_locked PUBLIC pinocchio::pinocchio Eigen3::Eigen)
add_library(fa_min_snap_locked STATIC vendor/min_snap/src/min_snap_trajectory.cpp vendor/min_snap/src/min_snap_planner.cpp)
target_include_directories(fa_min_snap_locked PUBLIC vendor/min_snap/include)
ament_target_dependencies(fa_min_snap_locked rclcpp sensor_msgs std_msgs)

add_library(fa_core STATIC src/core.cpp src/planner.cpp src/version.cpp src/transport.cpp)
target_include_directories(fa_core PUBLIC include ${CMAKE_CURRENT_BINARY_DIR}/bundle)
target_link_libraries(fa_core PUBLIC fa_ik_locked fa_min_snap_locked yaml-cpp OpenSSL::Crypto Threads::Threads ${ZMQ_LIBRARY})
target_compile_options(fa_core PRIVATE -Wall -Wextra -Wpedantic)

foreach(node teleop_node min_snap_node mock_robot)
  add_executable(fa_${node} src/${node}.cpp)
  target_link_libraries(fa_${node} fa_core "${typesupport}")
  ament_target_dependencies(fa_${node} rclcpp ament_index_cpp sensor_msgs std_msgs std_srvs)
endforeach()
add_executable(fa_core_tests tests/core_tests.cpp)
target_link_libraries(fa_core_tests fa_core)
add_executable(fa_benchmark tests/benchmark.cpp)
target_link_libraries(fa_benchmark fa_core)
add_executable(fa_fixture_dump tests/fixture_dump.cpp)
target_link_libraries(fa_fixture_dump fa_core)
enable_testing()
add_test(NAME core_contracts COMMAND fa_core_tests ${CMAKE_CURRENT_BINARY_DIR}/bundle)
add_test(NAME numerical_regression COMMAND fa_benchmark --bundle ${CMAKE_CURRENT_BINARY_DIR}/bundle --steps 1000 --output ${CMAKE_CURRENT_BINARY_DIR}/numerical.json)
add_test(NAME version_lock COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_version.py $<TARGET_FILE:fa_core_tests> ${CMAKE_CURRENT_BINARY_DIR}/bundle)
add_test(NAME python_mapping_parity COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_python_parity.py
  ${CMAKE_CURRENT_SOURCE_DIR}/../../../beavr-bot/src/beavr/teleop/components/detector/vr/keypoint_transform.py $<TARGET_FILE:fa_fixture_dump>)

install(TARGETS fa_teleop_node fa_min_snap_node fa_mock_robot fa_benchmark fa_core_tests RUNTIME DESTINATION lib/${PROJECT_NAME})
install(DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/bundle/ DESTINATION share/${PROJECT_NAME}/bundle)
install(DIRECTORY launch DESTINATION share/${PROJECT_NAME})
install(PROGRAMS tools/run_benchmark.py tools/verify_source.py tools/integration_test.py
  tools/beavr_state_bridge.py tools/create_receipt.py DESTINATION lib/${PROJECT_NAME})
ament_export_dependencies(rosidl_default_runtime)
ament_package()
