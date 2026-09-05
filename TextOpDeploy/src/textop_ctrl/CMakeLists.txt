cmake_minimum_required(VERSION 3.8)
project(textop_ctrl)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra)
  # Disable pedantic warnings for ONNX Runtime
  add_compile_options(-Wno-pedantic)
endif()

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(builtin_interfaces REQUIRED)
find_package(rosidl_default_generators REQUIRED)

# Find yaml-cpp
find_package(yaml-cpp REQUIRED)

# Find unitree message packages
find_package(unitree_hg REQUIRED)
find_package(unitree_go REQUIRED)

# Find cnpy for NPZ file loading
find_library(CNPY_LIBRARY cnpy PATHS "${CMAKE_CURRENT_SOURCE_DIR}/../../install/CNPY/lib")

# Find ONNX Runtime
# Use the existing ONNX Runtime installation

if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|amd64")
    set(ONNXRUNTIME_ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/onnxruntime-linux-x64-1.22.0" CACHE PATH "ONNX Runtime root directory")
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64")
    set(ONNXRUNTIME_ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/onnxruntime-linux-aarch64-1.22.0" CACHE PATH "ONNX Runtime root directory")
else()
    message(FATAL_ERROR "Unsupported architecture: ${CMAKE_SYSTEM_PROCESSOR}")
endif()


if(NOT EXISTS "${ONNXRUNTIME_ROOT_DIR}")
    message(FATAL_ERROR "ONNX Runtime not found at ${ONNXRUNTIME_ROOT_DIR}. Please set ONNXRUNTIME_ROOT_DIR to the correct path.")
endif()

# Set ONNX Runtime paths
set(ONNXRUNTIME_INCLUDE_DIRS "${ONNXRUNTIME_ROOT_DIR}/include")
set(ONNXRUNTIME_LIBRARIES "${ONNXRUNTIME_ROOT_DIR}/lib/libonnxruntime.so")

# Check if ONNX Runtime files exist
if(NOT EXISTS "${ONNXRUNTIME_INCLUDE_DIRS}/onnxruntime_cxx_api.h")
    message(FATAL_ERROR "ONNX Runtime headers not found at ${ONNXRUNTIME_INCLUDE_DIRS}")
endif()

# Include directories
include_directories(include)
include_directories(${ONNXRUNTIME_INCLUDE_DIRS})
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../../install/CNPY/include)

# Create executable
add_executable(textop_onnx_controller 
  src/textop_onnx_controller.cpp
  src/onnx_policy.cpp
  src/config_loader.cpp
  src/command_helper.cpp
  src/rotation_helper.cpp
  src/remote_controller.cpp
  src/motion_loader.cpp
  src/motor_crc_hg.cpp
  src/observation_computer.cpp
)

# Create motion block example executable
add_executable(motion_block_example
  src/motion_block_example.cpp
  src/motion_block_subscriber.cpp
)

# Link libraries
ament_target_dependencies(textop_onnx_controller
  rclcpp
  std_msgs
  geometry_msgs
  sensor_msgs
  unitree_hg
  unitree_go
  builtin_interfaces
)

# Add include directories for generated messages
target_include_directories(textop_onnx_controller PRIVATE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/rosidl_generator_cpp>
  $<INSTALL_INTERFACE:include>
)

target_link_libraries(textop_onnx_controller
  ${ONNXRUNTIME_LIBRARIES}
  yaml-cpp
  ${CNPY_LIBRARY}
  ${PROJECT_NAME}__rosidl_generator_c
  ${PROJECT_NAME}__rosidl_typesupport_cpp
)

# Link libraries for motion block example
ament_target_dependencies(motion_block_example
  rclcpp
  std_msgs
  builtin_interfaces
)

# Add include directories for generated messages
target_include_directories(motion_block_example PRIVATE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/rosidl_generator_cpp>
  $<INSTALL_INTERFACE:include>
)

# Link the generated message libraries
target_link_libraries(motion_block_example
  ${PROJECT_NAME}__rosidl_generator_c
  ${PROJECT_NAME}__rosidl_typesupport_cpp
  ${CNPY_LIBRARY}
  z  # zlib for cnpy
)

# Install
install(TARGETS
  textop_onnx_controller
  motion_block_example
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY
  launch
  config
  models
  include
  DESTINATION share/${PROJECT_NAME}/
)

# Generate messages
rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/MotionBlock.msg"
  DEPENDENCIES builtin_interfaces std_msgs
)

# Make sure the message generation is complete before building the executables
if(TARGET ${PROJECT_NAME})
  add_dependencies(textop_onnx_controller ${PROJECT_NAME})
  add_dependencies(motion_block_example ${PROJECT_NAME})
endif()

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
