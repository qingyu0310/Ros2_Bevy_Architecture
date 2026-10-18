# 视觉启动：相机适配 + 视觉节点，两节点都用仿真时钟。
# 用法：ros2 launch project vision.launch.py
#       ros2 launch project vision.launch.py use_sim_time:=false   # 接实机时

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, EmitEvent, RegisterEventHandler
from launch.event_handlers import OnProcessExit
from launch.events import Shutdown
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    use_sim_time = LaunchConfiguration('use_sim_time')

    # 仿真器（Daedalus）不发 /clock，用系统时间打时间戳；这里必须跟着用系统时间，
    # 否则节点 ROS 时间停在 0，帧龄/每秒统计全错
    use_sim_time_arg = DeclareLaunchArgument(
        'use_sim_time',
        default_value='false',
        description='仿真器不发 /clock，默认 false；接发布 /clock 的仿真或实机时再打开',
    )

    # 相机适配：/image_raw (rgb8) -> /vision/image (bgr8)，同时转发 /camera_info
    camera = Node(
        package='project',
        executable='camera',
        name='camera',
        output='screen',
        parameters=[{'use_sim_time': use_sim_time}],
    )

    # 视觉：检测、位姿、多目标跟踪、可视化
    vision = Node(
        package='project',
        executable='vision',
        name='vision',
        output='screen',
        parameters=[{'use_sim_time': use_sim_time}],
    )

    # 视觉节点退出时把相机节点一起收掉，避免留下只转发的空壳
    shutdown_on_vision_exit = RegisterEventHandler(
        OnProcessExit(
            target_action=vision,
            on_exit=[
                EmitEvent(event=Shutdown(reason='vision exited')),
            ],
        )
    )

    return LaunchDescription([use_sim_time_arg, camera, vision, shutdown_on_vision_exit])
