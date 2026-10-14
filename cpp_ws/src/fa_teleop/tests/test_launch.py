#!/usr/bin/env python3
import importlib.util
import sys
from launch import LaunchContext
spec = importlib.util.spec_from_file_location("fa_launch", sys.argv[1])
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
module.Node = lambda **kwargs: kwargs

def actions(backend, render="true", namespace="auto"):
    context = LaunchContext()
    context.launch_configurations.update(control_backend=backend, render=render, namespace=namespace,
                                         profile="precision_200hz", enable_motion="false", render_hz="30")
    return module.launch_nodes(context)

real = actions("real")
assert [a["executable"] for a in real] == ["fa_min_snap_node", "fa_teleop_node"]
assert all(a["namespace"] == "" and a["parameters"][0]["control_backend"] == "real" for a in real)
sim = actions("mujoco")
assert [a["executable"] for a in sim] == ["fa_mujoco_node", "fa_mujoco_viewer", "fa_min_snap_node", "fa_teleop_node"]
assert all(a["namespace"] == "fa_mujoco" for a in sim)
assert "fa_mujoco_viewer" not in [a["executable"] for a in actions("mujoco", "false")]
for backend, namespace in [("invalid", "auto"), ("mujoco", "/")]:
    try:
        actions(backend, namespace=namespace)
    except ValueError:
        pass
    else:
        raise AssertionError((backend, namespace))
print("PASS real/mujoco routing, default namespace, headless mode, invalid-mode rejection")
