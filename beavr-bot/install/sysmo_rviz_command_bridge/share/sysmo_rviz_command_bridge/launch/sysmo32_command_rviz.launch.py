from pathlib import Path

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
    return text.replace('filename="../meshes/', f'filename="{mesh_dir_uri}/')


def _launch_setup(context, *args, **kwargs):
    robot_description = _load_robot_description(LaunchConfiguration("urdf_path").perform(context))
    command_topic = LaunchConfiguration("command_topic")
    joint_state_topic = LaunchConfiguration("joint_state_topic")
    interpolation_steps = LaunchConfiguration("interpolation_steps")
    publish_rate_hz = LaunchConfiguration("publish_rate_hz")
    use_robot_state_publisher = LaunchConfiguration("use_robot_state_publisher")
    use_rviz = LaunchConfiguration("use_rviz")
    rviz_config = LaunchConfiguration("rviz_config")

    return [
        Node(
            package="robot_state_publisher",
            executable="robot_state_publisher",
            output="screen",
            condition=IfCondition(use_robot_state_publisher),
            parameters=[{"robot_description": robot_description}],
            remappings=[("joint_states", joint_state_topic)],
        ),
        Node(
            package="sysmo_rviz_command_bridge",
            executable="sysmo32_command_to_joint_state",
            output="screen",
            parameters=[
                {
                    "command_topic": command_topic,
                    "joint_state_topic": joint_state_topic,
                    "interpolation_steps": interpolation_steps,
                    "publish_rate_hz": publish_rate_hz,
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
            FindPackageShare("sysmo_rviz_command_bridge"),
            "rviz",
            "sysmo32_command.rviz",
        ]
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "urdf_path",
                default_value="/home/likunwei/dataCollection/beavr-bot/robots/sysmo_description/urdf/sysmo32.urdf",
                description="Path to the SYSMO-32 arm URDF file.",
            ),
            DeclareLaunchArgument(
                "command_topic",
                default_value="/sysmo_left_arm_controller/commands",
                description="18D Float64MultiArray command topic from the teleop real interface.",
            ),
            DeclareLaunchArgument(
                "joint_state_topic",
                default_value="/joint_states",
                description="JointState feedback topic used by robot_state_publisher.",
            ),
            DeclareLaunchArgument(
                "interpolation_steps",
                default_value="5",
                description="Number of quintic polynomial interpolation samples for every incoming command.",
            ),
            DeclareLaunchArgument(
                "publish_rate_hz",
                default_value="100.0",
                description="Rate used to publish interpolated JointState samples.",
            ),
            DeclareLaunchArgument(
                "use_robot_state_publisher",
                default_value="true",
                description="Launch robot_state_publisher with the SYSMO-32 model.",
            ),
            DeclareLaunchArgument(
                "use_rviz",
                default_value="true",
                description="Launch rviz2 for visualizing the robot model.",
            ),
            DeclareLaunchArgument(
                "rviz_config",
                default_value=rviz_default,
                description="RViz configuration with RobotModel already enabled.",
            ),
            OpaqueFunction(function=_launch_setup),
        ]
    )
