from pathlib import Path
import re

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def _load_robot_description(urdf_path: str) -> str:
    path = Path(urdf_path).expanduser().resolve()
    text = path.read_text()
    mesh_dir_uri = path.parent.parent.joinpath("meshes").resolve().as_uri()
    text = re.sub(
        r'filename="package://(?:sysmo_description|fa_description)/meshes/([^"]+)"',
        rf'filename="{mesh_dir_uri}/\1"',
        text,
    )
    return text.replace('filename="../meshes/', f'filename="{mesh_dir_uri}/')


def _launch_setup(context, *args, **kwargs):
    robot_description = _load_robot_description(LaunchConfiguration("urdf_path").perform(context))
    command_topic = LaunchConfiguration("command_topic")
    joint_state_topic = LaunchConfiguration("joint_state_topic")
    interpolation_steps = LaunchConfiguration("interpolation_steps")
    interpolation_profile = LaunchConfiguration("interpolation_profile")
    publish_rate_hz = LaunchConfiguration("publish_rate_hz")
    idle_publish_rate_hz = LaunchConfiguration("idle_publish_rate_hz")
    use_command_bridge = LaunchConfiguration("use_command_bridge")
    use_robot_state_publisher = LaunchConfiguration("use_robot_state_publisher")
    publish_base_tf = LaunchConfiguration("publish_base_tf")
    world_frame = LaunchConfiguration("world_frame")
    base_frame = LaunchConfiguration("base_frame")
    use_rviz = LaunchConfiguration("use_rviz")
    rviz_config = LaunchConfiguration("rviz_config")

    return [
        Node(
            package="tf2_ros",
            executable="static_transform_publisher",
            output="screen",
            condition=IfCondition(publish_base_tf),
            arguments=["--frame-id", world_frame, "--child-frame-id", base_frame],
        ),
        Node(
            package="robot_state_publisher",
            executable="robot_state_publisher",
            output="screen",
            condition=IfCondition(use_robot_state_publisher),
            parameters=[{"robot_description": robot_description}],
            remappings=[("joint_states", joint_state_topic)],
        ),
        Node(
            package="fa_rviz_command_bridge",
            executable="fa_command_to_joint_state",
            output="screen",
            condition=IfCondition(use_command_bridge),
            parameters=[
                {
                    "command_topic": command_topic,
                    "joint_state_topic": joint_state_topic,
                    "interpolation_steps": interpolation_steps,
                    "interpolation_profile": interpolation_profile,
                    "publish_rate_hz": publish_rate_hz,
                    "idle_publish_rate_hz": idle_publish_rate_hz,
                    "expected_command_length": 16,
                }
            ],
        ),
        Node(
            package="rviz2",
            executable="rviz2",
            output="screen",
            condition=IfCondition(use_rviz),
            arguments=["-d", rviz_config],
        ),
    ]


def generate_launch_description():
    rviz_default = PathJoinSubstitution(
        [
            FindPackageShare("fa_rviz_command_bridge"),
            "rviz",
            "fa_command.rviz",
        ]
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "urdf_path",
                default_value="/home/likunwei/dataCollection/beavr-bot/robots/fa_description/urdf/fa_robot.urdf",
                description="Path to the FA robot URDF file.",
            ),
            DeclareLaunchArgument(
                "command_topic",
                default_value="/upper_position_controller/commands",
                description="16D Float64MultiArray command topic from the FA teleop interface.",
            ),
            DeclareLaunchArgument(
                "joint_state_topic",
                default_value="/joint_states",
                description="JointState feedback topic used by robot_state_publisher.",
            ),
            DeclareLaunchArgument(
                "interpolation_steps",
                default_value="5",
                description="Number of polynomial interpolation samples for every incoming command.",
            ),
            DeclareLaunchArgument(
                "interpolation_profile",
                default_value="min_snap",
                description="Command bridge interpolation profile: min_snap or quintic.",
            ),
            DeclareLaunchArgument(
                "publish_rate_hz",
                default_value="1000.0",
                description="Rate used to publish interpolated JointState samples.",
            ),
            DeclareLaunchArgument(
                "idle_publish_rate_hz",
                default_value="1000.0",
                description="Rate used to keep publishing JointState when no command is active.",
            ),
            DeclareLaunchArgument(
                "use_command_bridge",
                default_value="true",
                description="Launch the FA command-to-JointState bridge.",
            ),
            DeclareLaunchArgument(
                "use_robot_state_publisher",
                default_value="true",
                description="Launch robot_state_publisher with the FA model.",
            ),
            DeclareLaunchArgument(
                "publish_base_tf",
                default_value="true",
                description="Publish an identity static transform from world_frame to base_frame.",
            ),
            DeclareLaunchArgument(
                "world_frame",
                default_value="world",
                description="RViz world frame used as the root visualization frame.",
            ),
            DeclareLaunchArgument(
                "base_frame",
                default_value="pelvis",
                description="FA root link frame. The URDF root link is pelvis.",
            ),
            DeclareLaunchArgument(
                "use_rviz",
                default_value="true",
                description="Launch rviz2 for visualizing the FA model.",
            ),
            DeclareLaunchArgument(
                "rviz_config",
                default_value=rviz_default,
                description="RViz configuration with RobotModel already enabled.",
            ),
            OpaqueFunction(function=_launch_setup),
        ]
    )
