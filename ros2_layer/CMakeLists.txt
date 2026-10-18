cmake_minimum_required(VERSION 3.8)
project(ros2_layer)

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(tf2_msgs REQUIRED)
find_package(rm_interfaces REQUIRED)
find_package(OpenCV REQUIRED COMPONENTS core imgproc highgui)
find_package(framework REQUIRED)

# ============================================================================
#
# 节点类
#
# 节点写成 .hpp 声明 + .cpp 实现，编成静态库导出。消费方（project）写
#   #include "ros2_layer/node/vision/vision_node.hpp"
#   rclcpp::spin(std::make_shared<VisionNode>());
# rclcpp / 消息包 / framework 都从这里透传下去，project 不用自己 find_package
#
# ============================================================================

add_library(${PROJECT_NAME} STATIC
    node/vision/vision_node.cpp
    node/vision/aim_publisher.cpp)
target_include_directories(${PROJECT_NAME} PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/..>
    $<INSTALL_INTERFACE:include>)
target_compile_features(${PROJECT_NAME} PUBLIC cxx_std_17)
target_link_libraries(${PROJECT_NAME} PUBLIC
    rclcpp::rclcpp
    ${sensor_msgs_TARGETS}
    ${std_msgs_TARGETS}
    ${geometry_msgs_TARGETS}
    opencv_core opencv_imgproc opencv_highgui
    tf2_ros::tf2_ros
    ${tf2_msgs_TARGETS}
    rm_interfaces::rm_interfaces__rosidl_typesupport_cpp
    framework::framework_vision)

# ============================================================================
#
# 安装
#
# 头装成 include/ros2_layer/node/...，配合下面的 ament_export_include_directories
# 消费方才能写 #include "ros2_layer/node/vision/vision_node.hpp"。
# FILES_MATCHING 只挑 .hpp：本包对外只暴露头文件
#
# ============================================================================

install(DIRECTORY node
    DESTINATION include/ros2_layer
    FILES_MATCHING PATTERN "*.hpp")

install(TARGETS ${PROJECT_NAME} EXPORT ${PROJECT_NAME}_targets)
install(EXPORT ${PROJECT_NAME}_targets
    FILE ${PROJECT_NAME}Targets.cmake
    NAMESPACE ${PROJECT_NAME}::
    DESTINATION share/${PROJECT_NAME}/cmake)

ament_export_include_directories(include)
ament_export_targets(${PROJECT_NAME}_targets)
ament_export_dependencies(rclcpp sensor_msgs std_msgs geometry_msgs tf2_ros tf2_msgs rm_interfaces framework OpenCV)

ament_package()
