#!/usr/bin/python3
"""Exercise the public launch command with C++ MuJoCo on isolated ROS topics."""
import argparse
import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import re
import subprocess
import time


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--prefix", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--seconds", type=float, default=20)
    args = p.parse_args()
    os.environ.update(ROS_LOCALHOST_ONLY="1", ROS_DOMAIN_ID="196")
    import rclpy
    from rclpy.qos import QoSProfile, ReliabilityPolicy
    from sensor_msgs.msg import JointState
    from std_msgs.msg import Float64MultiArray, String
    import zmq
    args.output.mkdir(parents=True, exist_ok=True)
    bundle = args.prefix.resolve() / "share/fa_teleop/bundle"
    release = hashlib.sha256((bundle / "release.json").read_bytes()).hexdigest()
    identifier = release + ":" + hashlib.sha256((bundle / "config/simulation_test.yaml").read_bytes()).hexdigest() + ":mujoco"
    rclpy.init()
    node = rclpy.create_node("mujoco_test_observer", namespace="/fa_cpp_mujoco_test")
    observation = {"backend": "", "teleop": {}, "executor": "", "feedback_count": 0, "commands": 0,
                   "feedback": None, "first_feedback": None, "max_joint_motion_rad": 0,
                   "last_command": None, "max_joint_step_rad": 0, "root_feedback": 0, "root_commands": 0}
    sensor_qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT)
    status_samples = []

    def teleop_status(msg):
        status = json.loads(msg.data)
        observation["teleop"] = status
        status_samples.append((time.monotonic(), status["cycles"]))

    def feedback(msg):
        assert len(msg.name) == len(msg.position) == len(msg.velocity) == 16
        assert msg.name[-2:] == ["neck_yaw_joint", "neck_pitch_joint"]
        assert all(math.isfinite(x) for x in [*msg.position, *msg.velocity])
        observation["feedback"] = msg
        observation["feedback_count"] += 1
        if observation["first_feedback"] is None:
            observation["first_feedback"] = list(msg.position)
        observation["max_joint_motion_rad"] = max(observation["max_joint_motion_rad"], max(abs(a-b) for a, b in zip(observation["first_feedback"], msg.position)))

    def command(msg):
        assert len(msg.data) == 16
        if observation["last_command"] is not None:
            observation["max_joint_step_rad"] = max(observation["max_joint_step_rad"], max(abs(a-b) for a, b in zip(observation["last_command"], msg.data)))
        observation["last_command"] = list(msg.data)
        observation["commands"] += 1

    subscriptions = [
        node.create_subscription(JointState, "joint_states", feedback, sensor_qos),
        node.create_subscription(Float64MultiArray, "upper_position_controller/commands", command, 1),
        node.create_subscription(String, "fa_cpp/backend_status", lambda m: observation.update(backend=m.data), 1),
        node.create_subscription(String, "fa_cpp/executor_status", lambda m: observation.update(executor=m.data), 1),
        node.create_subscription(String, "fa_cpp/teleop_status", teleop_status, 10),
        node.create_subscription(JointState, "/joint_states", lambda m: observation.update(root_feedback=observation["root_feedback"]+1), sensor_qos),
        node.create_subscription(Float64MultiArray, "/upper_position_controller/commands", lambda m: observation.update(root_commands=observation["root_commands"]+1), 1),
    ]
    context = zmq.Context()
    sockets = []
    for port in [18087, 18110, 18100]:
        socket = context.socket(zmq.PUSH)
        socket.setsockopt(zmq.LINGER, 0)
        socket.setsockopt(zmq.SNDHWM, 2)
        socket.connect(f"tcp://127.0.0.1:{port}")
        sockets.append(socket)
    log = open(args.output / "launch.log", "w")
    proc = subprocess.Popen(["ros2", "launch", "fa_teleop", "teleop.launch.py", "control_backend:=mujoco",
                             "render:=false", "namespace:=fa_cpp_mujoco_test", "profile:=simulation_test", "enable_motion:=true"],
                            stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    last_send = 0
    motion_start = None

    def send_hands():
        nonlocal last_send
        now = time.monotonic()
        if now-last_send < 1/60:
            return
        last_send = now
        shift = 0 if motion_start is None else 0.002 * math.sin((now-motion_start)*0.5)
        stamp = datetime.datetime.now().strftime("%H:%M:%S.%f")
        for index, socket in enumerate(sockets[:2]):
            wrist = [(-0.2 if index == 1 else 0.2) + shift, 0, -0.3]
            points = [list(wrist) for _ in range(26)]
            for k, d in [(7, [.03, .08, 0]), (12, [0, .09, 0]), (22, [-.03, .07, 0])]:
                points[k] = [a+b for a, b in zip(wrist, d)]
            raw = stamp + ":relative:" + "|".join(",".join(str(x) for x in q) for q in points) + ":hand_command=1"
            try:
                socket.send_string(raw, zmq.NOBLOCK)
            except zmq.Again:
                pass

    def spin(seconds, hands=True, tracking=False):
        end = time.monotonic()+seconds
        while time.monotonic() < end:
            assert proc.poll() is None, f"Launch exited; see {args.output}/launch.log"
            if hands:
                send_hands()
            rclpy.spin_once(node, timeout_sec=.002)
            if tracking:
                assert observation["teleop"].get("stage", 3) == 3, observation["teleop"]
            assert "|FAULT" not in observation["backend"], observation["backend"]
            assert "|FAULT" not in observation["executor"], observation["executor"]

    try:
        spin(3)
        assert observation["backend"].startswith(identifier+"|READY"), observation["backend"]
        assert observation["feedback_count"] > 200
        assert observation["commands"] == 0, "Simulation executed before PICO enable"
        assert max(abs(x) for x in observation["feedback"].position) < 1e-12
        sockets[2].send_string("High")
        until = time.monotonic()+16
        while observation["teleop"].get("stage") != 3 and time.monotonic() < until:
            spin(.1)
        assert observation["teleop"].get("stage") == 3, observation["teleop"]
        assert observation["max_joint_motion_rad"] > .9, "Initialization trajectory was not executed by MuJoCo"
        motion_start = time.monotonic()
        first_tracking_sample = len(status_samples)
        feedback_before = observation["feedback_count"]
        backend_before = int(re.search(r"\|feedback=(\d+)", observation["backend"]).group(1))
        start = time.monotonic()
        spin(args.seconds, tracking=True)
        elapsed = time.monotonic()-start
        # Use both counter and time from the same diagnostic sample boundaries.
        first_status, last_status = status_samples[first_tracking_sample], status_samples[-1]
        control_hz = (last_status[1]-first_status[1])/(last_status[0]-first_status[0])
        feedback_hz = (observation["feedback_count"]-feedback_before)/elapsed
        backend_hz = (int(re.search(r"\|feedback=(\d+)", observation["backend"]).group(1))-backend_before)/elapsed
        assert 195 <= control_hz <= 205, control_hz
        sockets[2].send_string("Low")
        spin(1.3)
        assert observation["teleop"].get("stage") == 4
        sockets[2].send_string("High")
        spin(2)
        assert observation["teleop"].get("stage") == 3
        spin(1.5, hands=False)
        assert observation["teleop"].get("stage") == 4
        assert observation["root_feedback"] == observation["root_commands"] == 0
        assert observation["max_joint_step_rad"] < .01
        report = {k:v for k,v in observation.items() if k not in {"feedback", "first_feedback", "last_command"}}
        report.update(passed=True, release_id=release, engine="MuJoCo 3.8.0 C API", physical_hardware=False,
                      tracking_seconds=elapsed, approximate_control_hz=control_hz, observed_feedback_hz=feedback_hz,
                      approximate_backend_feedback_hz=backend_hz,
                      command=["ros2 launch fa_teleop teleop.launch.py", "control_backend:=mujoco", "render:=false"])
        (args.output / "result.json").write_text(json.dumps(report, indent=2)+"\n")
        print(json.dumps(report, indent=2))
    finally:
        if proc.poll() is None:
            os.killpg(proc.pid, signal.SIGINT)
        try:
            proc.wait(timeout=8)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGTERM)
            proc.wait(timeout=5)
        log.close()
        for socket in sockets:
            socket.close()
        context.term()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
