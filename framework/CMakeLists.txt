cmake_minimum_required(VERSION 3.8)
project(framework)

find_package(ament_cmake REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(OpenCV REQUIRED COMPONENTS core imgproc calib3d dnn imgcodecs)

# CUDA / TensorRT：检测器唯一的外部重依赖。不是 ROS 包，直接找。
# 同一份逻辑也装给下游用，见文件末尾的 ament_package(CONFIG_EXTRAS ...)
include(cmake/dependencies.cmake)

# ============================================================================
#
# vision：视觉算法核心库
#
# 纯计算，不依赖 ROS、线程与设备；源文件全部显式登记，不做目录扫描
#
# ============================================================================

add_library(framework_vision STATIC
    algorithm/vision/core/detections.cpp
    algorithm/vision/core/light_bar.cpp
    algorithm/vision/detect/model_decode.cpp
    algorithm/vision/detect/armor_detector.cpp
    algorithm/vision/pose/armor_pose.cpp
    algorithm/vision/track/association.cpp
    algorithm/vision/track/vehicle_ekf.cpp
    algorithm/vision/track/tracker.cpp
    algorithm/vision/aim/aim_solver.cpp
    algorithm/vision/refine/armor_refiner.cpp
    algorithm/calib/camera_calib.cpp)

target_include_directories(framework_vision PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/..>
    $<INSTALL_INTERFACE:include>)
target_link_libraries(framework_vision PUBLIC
    Eigen3::Eigen
    opencv_core opencv_imgproc opencv_calib3d opencv_dnn opencv_imgcodecs)
# 静态库：TensorRT/CUDA 的符号要留给最终可执行文件解析，所以必须是 PUBLIC
target_link_libraries(framework_vision PUBLIC CUDA::cudart TensorRT::nvinfer)
target_compile_features(framework_vision PUBLIC cxx_std_17)
target_compile_options(framework_vision PRIVATE -Wall -Wextra -Wpedantic)

# ============================================================================
#
# 安装
#
# 头文件装成 include/framework/algorithm/...，消费方写
# #include "framework/algorithm/vision/..."
#
# ============================================================================

install(DIRECTORY algorithm
    DESTINATION include/framework
    FILES_MATCHING PATTERN "*.hpp")

install(TARGETS framework_vision
    EXPORT ${PROJECT_NAME}_targets
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib)

install(EXPORT ${PROJECT_NAME}_targets
    FILE ${PROJECT_NAME}Targets.cmake
    NAMESPACE ${PROJECT_NAME}::
    DESTINATION share/${PROJECT_NAME}/cmake)

ament_export_include_directories(include)
ament_export_targets(${PROJECT_NAME}_targets)
ament_export_dependencies(Eigen3 OpenCV)

ament_package(CONFIG_EXTRAS cmake/dependencies.cmake)
