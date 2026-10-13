from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    enabled = ParameterValue(LaunchConfiguration("enable_motion"), value_type=bool)
    profile = LaunchConfiguration("profile")
    namespace = LaunchConfiguration("namespace")
    return LaunchDescription([
        DeclareLaunchArgument("enable_motion", default_value="false"),
        DeclareLaunchArgument("profile", default_value="precision_200hz"),
        DeclareLaunchArgument("namespace", default_value=""),
        Node(package="fa_teleop", executable="fa_min_snap_node", namespace=namespace,
             parameters=[{"enable_motion": enabled, "profile": profile}], output="screen"),
        Node(package="fa_teleop", executable="fa_teleop_node", namespace=namespace,
             parameters=[{"enable_motion": enabled, "profile": profile}], output="screen"),
    ])
