from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction, Shutdown
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def launch_nodes(context):
    backend = LaunchConfiguration("control_backend").perform(context)
    if backend not in {"real", "mujoco"}:
        raise ValueError("control_backend must be real or mujoco")
    namespace = LaunchConfiguration("namespace").perform(context).strip("/")
    if namespace == "auto":
        namespace = "fa_mujoco" if backend == "mujoco" else ""
    if backend == "mujoco" and not namespace:
        raise ValueError("MuJoCo mode requires an isolated namespace; use namespace:=auto")
    params = {"control_backend": backend,
              "enable_motion": ParameterValue(LaunchConfiguration("enable_motion"), value_type=bool),
              "profile": LaunchConfiguration("profile")}
    def component(executable, extra=None):
        return Node(package="fa_teleop", executable=executable, namespace=namespace,
                    parameters=[params, extra or {}], output="screen",
                    on_exit=Shutdown(reason=executable + " exited"))
    nodes = []
    if backend == "mujoco":
        nodes.append(component("fa_mujoco_node"))
        render = LaunchConfiguration("render").perform(context).lower()
        if render not in {"true", "false"}:
            raise ValueError("render must be true or false")
        if render == "true":
            nodes.append(component("fa_mujoco_viewer", {
                "render_hz": ParameterValue(LaunchConfiguration("render_hz"), value_type=float)}))
    nodes += [component("fa_min_snap_node"), component("fa_teleop_node")]
    return nodes


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument("control_backend", default_value="mujoco", choices=["real", "mujoco"]),
        DeclareLaunchArgument("enable_motion", default_value="false", choices=["true", "false"]),
        DeclareLaunchArgument("profile", default_value="precision_200hz"),
        DeclareLaunchArgument("namespace", default_value="auto"),
        DeclareLaunchArgument("render", default_value="true", choices=["true", "false"]),
        DeclareLaunchArgument("render_hz", default_value="30.0"),
        OpaqueFunction(function=launch_nodes),
    ])
