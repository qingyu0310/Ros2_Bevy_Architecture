cmake_minimum_required(VERSION 3.8)
project(project)

find_package(ament_cmake REQUIRED)
find_package(ament_index_cpp REQUIRED)
find_package(rclcpp REQUIRED)
find_package(ros2_layer REQUIRED)
# CMAKE_INSTALL_RPATH_USE_LINK_PATH 引用的 imported target
find_package(CUDAToolkit REQUIRED)

# CUDA / TensorRT 只装在 .venv 里（非系统路径），把链接时用到的库目录写进 RPATH，
# 运行时不依赖 LD_LIBRARY_PATH
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH ON)

# ============================================================================
#
# 节点入口
#
# 节点本体在 ros2_layer/node/*/*.hpp，这里只有 main。
# 链 ros2_layer::ros2_layer 就够了：rclcpp / 消息包 / framework（含算法库与 Eigen）都由它透传
#
# ============================================================================

add_executable(camera node/camera/camera_node.cpp)
target_link_libraries(camera ros2_layer::ros2_layer ament_index_cpp::ament_index_cpp)
install(TARGETS camera DESTINATION lib/${PROJECT_NAME})

add_executable(vision node/vision/vision_node.cpp)
target_link_libraries(vision ros2_layer::ros2_layer ament_index_cpp::ament_index_cpp)
install(TARGETS vision DESTINATION lib/${PROJECT_NAME})

# ============================================================================
#
# 参数与 launch
#
# 参数文件按节点名分段（/camera:、/vision:），入口从 share/project/params/ 下加载
#
# ============================================================================

install(DIRECTORY params launch
    DESTINATION share/${PROJECT_NAME})

ament_package()
