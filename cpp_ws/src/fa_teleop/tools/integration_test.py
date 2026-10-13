#!/usr/bin/env python3
"""Run actual C++ nodes on a local-only ROS domain and isolated namespace."""
import argparse
import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import time


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--bin-dir", type=Path, required=True)
    p.add_argument("--bundle", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    os.environ["ROS_LOCALHOST_ONLY"] = "1"
    os.environ["ROS_DOMAIN_ID"] = "197"
    import rclpy
    from rclpy.qos import QoSProfile, ReliabilityPolicy
    from sensor_msgs.msg import JointState
    from std_msgs.msg import Float64MultiArray, String
    from std_srvs.srv import Trigger
    from fa_teleop.msg import JointTarget
    import zmq

    namespace = "/fa_cpp_integration"
    args.output.mkdir(parents=True, exist_ok=True)
    release_id = hashlib.sha256((args.bundle / "release.json").read_bytes()).hexdigest()
    release_id += ":" + hashlib.sha256((args.bundle / "config/simulation_test.yaml").read_bytes()).hexdigest()
    processes, logs = [], []

    def start(executable, extra=()):
        log = open(args.output / f"{executable}.log", "w")
        logs.append(log)
        cmd = [str(args.bin_dir / executable), "--ros-args", "-r", f"__ns:={namespace}",
               "-p", f"bundle_dir:={args.bundle}", "-p", "enable_motion:=true", *extra]
        processes.append(subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, start_new_session=True))

    rclpy.init()
    node = rclpy.create_node("fa_integration_observer", namespace=namespace)
    sensor_qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT)
    observations = {"commands": 0, "executor": "", "teleop": {}, "last_command": None,
                    "feedback": None, "max_step_rad": 0.0}

    def command(msg):
        assert len(msg.data) == 16 and all(math.isfinite(x) for x in msg.data)
        previous = observations["last_command"]
        if previous:
            observations["max_step_rad"] = max(observations["max_step_rad"], max(abs(a-b) for a, b in zip(previous, msg.data)))
        observations["last_command"] = list(msg.data)
        observations["commands"] += 1

    subscriptions = [
        node.create_subscription(Float64MultiArray, "upper_position_controller/commands", command, 1),
        node.create_subscription(JointState, "joint_states", lambda m: observations.update(feedback=m), sensor_qos),
        node.create_subscription(String, "fa_cpp/executor_status", lambda m: observations.update(executor=m.data), 1),
        node.create_subscription(String, "fa_cpp/teleop_status", lambda m: observations.update(teleop=json.loads(m.data)), 10),
    ]
    publisher = node.create_publisher(JointTarget, "fa_cpp/target", 1)
    context = zmq.Context()
    sockets = []
    try:
        start("fa_mock_robot")
        start("fa_min_snap_node", ["-p", "profile:=simulation_test"])

        def spin_for(seconds, send=None):
            until = time.monotonic() + seconds
            while time.monotonic() < until:
                rclpy.spin_once(node, timeout_sec=0.002)
                if send:
                    send()
                for process in processes:
                    if process.poll() is not None:
                        raise RuntimeError(f"Test process exited: {process.args[0]}; see {args.output}")

        spin_for(2)
        assert observations["feedback"] is not None, "No mock feedback"
        assert observations["executor"].startswith(release_id + "|READY"), "No version-matched executor"
        assert observations["commands"] == 0, "Executor moved before explicit initialization"
        q = list(observations["feedback"].position)
        sequence = 0

        def send_target(mode=1, mismatch=False, stale=False):
            nonlocal sequence
            sequence += 1
            m = JointTarget()
            m.header.stamp = node.get_clock().now().to_msg()
            if stale:
                m.header.stamp.sec -= 5
            m.header.frame_id = "pelvis"
            m.release_id = "wrong-release" if mismatch else release_id
            m.sequence = sequence
            m.mode = mode
            m.left, m.right, m.neck = q[:7], q[7:14], q[14:]
            m.duration_s, m.velocity_limit, m.acceleration_limit = 0.04, 0.5, 3.0
            publisher.publish(m)

        spin_for(0.5, lambda: send_target(stale=True))
        assert observations["commands"] == 0, "Stale initialization accepted"
        spin_for(1, send_target)
        assert observations["commands"] > 200, "No trajectory sampling"
        send_target(mismatch=True)
        spin_for(0.3)
        assert "|FAULT" in observations["executor"], "Version mismatch did not latch fault"
        before = observations["commands"]
        spin_for(0.3)
        assert observations["commands"] == before, "Fault continued issuing commands"
        reset = node.create_client(Trigger, "fa_cpp/executor_reset")
        assert reset.wait_for_service(timeout_sec=2)
        future = reset.call_async(Trigger.Request())
        rclpy.spin_until_future_complete(node, future, timeout_sec=2)
        assert future.result().success
        spin_for(0.3)

        start("fa_teleop_node", ["-p", "profile:=simulation_test"])
        for port in [18087, 18110, 18100]:
            socket = context.socket(zmq.PUSH)
            socket.setsockopt(zmq.LINGER, 0)
            socket.setsockopt(zmq.SNDHWM, 2)
            socket.connect(f"tcp://127.0.0.1:{port}")
            sockets.append(socket)
        started = time.monotonic()
        move = False

        def hands():
            t = time.monotonic() - started
            shift = 0.001 * math.sin(t * 0.5) if move else 0.0
            stamp = datetime.datetime.now().strftime("%H:%M:%S.%f")
            for index, socket in enumerate(sockets[:2]):
                wrist = [(-0.2 if index == 1 else 0.2) + shift, 0, -0.3]
                points = [list(wrist) for _ in range(26)]
                for k, delta in [(7, [0.03, 0.08, 0]), (12, [0, 0.09, 0]), (22, [-0.03, 0.07, 0])]:
                    points[k] = [a+b for a, b in zip(wrist, delta)]
                payload = stamp + ":relative:" + "|".join(",".join(str(x) for x in point) for point in points) + ":hand_command=1"
                try:
                    socket.send_string(payload, zmq.NOBLOCK)
                except zmq.Again:
                    pass

        spin_for(1.5, hands)
        sockets[2].send_string("High")
        spin_for(3, hands)
        assert observations["teleop"].get("stage") == 3, observations["teleop"]
        move = True
        before = observations["commands"]
        spin_for(3, hands)
        assert observations["teleop"].get("stage") == 3, observations["teleop"]
        assert observations["commands"] - before > 500, "Tracking stream stalled"
        sockets[2].send_string("Low")
        spin_for(1.5, hands)
        assert observations["teleop"].get("stage") == 4, observations["teleop"]
        sockets[2].send_string("High")
        spin_for(2, hands)
        assert observations["teleop"].get("stage") == 3, observations["teleop"]
        spin_for(2)
        assert observations["teleop"].get("stage") == 4, "VR loss did not pause tracking"
        assert observations["max_step_rad"] < 0.01, "Discontinuous output"
        report = {k: v for k, v in observations.items() if k not in {"feedback", "last_command"}}
        report.update(passed=True, namespace=namespace, domain_id=197, physical_hardware=False)
        (args.output / "result.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report, indent=2))
    finally:
        for process in reversed(processes):
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGINT)
        for process in processes:
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGTERM)
                process.wait(timeout=5)
        for socket in sockets:
            socket.close()
        context.term()
        node.destroy_node()
        rclpy.shutdown()
        for log in logs:
            log.close()


if __name__ == "__main__":
    main()
