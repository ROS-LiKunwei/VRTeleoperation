#!/usr/bin/python3
"""Independent black-box tests for ownership, feedback and configuration guards."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--prefix", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--backend", choices=["real", "mujoco"], default="real")
    args = p.parse_args()
    os.environ.update(ROS_LOCALHOST_ONLY="1", ROS_DOMAIN_ID="198")
    import rclpy
    from rclpy.qos import QoSProfile, ReliabilityPolicy
    from fa_teleop.msg import JointTarget
    from sensor_msgs.msg import JointState
    from std_msgs.msg import String, Float64MultiArray
    from std_srvs.srv import Trigger
    root = args.prefix.resolve()
    bundle = root / "share/fa_teleop/bundle"
    identifier = hashlib.sha256((bundle / "release.json").read_bytes()).hexdigest() + ":" + hashlib.sha256((bundle / "config/simulation_test.yaml").read_bytes()).hexdigest()
    identifier += ":" + args.backend
    args.output.mkdir(parents=True, exist_ok=True)
    rclpy.init()
    node = rclpy.create_node("guard_observer", namespace="/fa_guard_test")
    sensor_qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT)
    feedback = node.create_publisher(JointState, "joint_states", sensor_qos)
    targets = node.create_publisher(JointTarget, "fa_cpp/target", 1)
    backend_status = node.create_publisher(String, "fa_cpp/backend_status", 1)
    state = {"status": "", "commands": 0}
    subs = [node.create_subscription(String, "fa_cpp/executor_status", lambda m: state.update(status=m.data), 1),
            node.create_subscription(Float64MultiArray, "upper_position_controller/commands", lambda m: state.update(commands=state["commands"]+1), 1)]
    q = [-1.05, .76, -.62, -1.03, -.68, 0, -.35, -1.05, -.76, .62, -1.03, .68, 0, .35, 0, 0]
    q = [float(value) for value in q]
    suffixes = ["shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow", "wrist_yaw", "wrist_pitch", "wrist_roll"]
    names = [f"{side}_{name}_joint" for side in ["left", "right"] for name in suffixes] + ["neck_yaw_joint", "neck_pitch_joint"]
    sequence = 0
    log = open(args.output / "executor.log", "w")
    proc = subprocess.Popen([str(root / "lib/fa_teleop/fa_min_snap_node"), "--ros-args", "-r", "__ns:=/fa_guard_test", "-p", "enable_motion:=true", "-p", "profile:=simulation_test", "-p", f"control_backend:={args.backend}", "-p", f"bundle_dir:={bundle}"], stdout=log, stderr=subprocess.STDOUT, start_new_session=True)

    def spin(seconds, send_target=False, send_feedback=True, wrong_profile=False, send_backend=True, wrong_backend=False):
        nonlocal sequence
        end = time.monotonic() + seconds
        while time.monotonic() < end:
            assert proc.poll() is None, "Executor process failed"
            if args.backend == "mujoco" and send_backend:
                m = String()
                m.data = (identifier if not wrong_backend else "wrong-release:mujoco") + "|READY"
                backend_status.publish(m)
            if send_feedback:
                m = JointState()
                m.header.stamp = node.get_clock().now().to_msg()
                m.name, m.position, m.velocity = names, q, [0.0]*16
                feedback.publish(m)
            if send_target:
                sequence += 1
                m = JointTarget()
                m.header.stamp = node.get_clock().now().to_msg()
                m.header.frame_id = "pelvis"
                m.sequence, m.mode = sequence, 1
                m.release_id = identifier if not wrong_profile else identifier.split(":")[0] + ":wrong-profile"
                m.left, m.right, m.neck = q[:7], q[7:14], q[14:]
                m.duration_s, m.velocity_limit, m.acceleration_limit = .04, .5, 3.0
                targets.publish(m)
            rclpy.spin_once(node, timeout_sec=.002)

    def reset():
        client = node.create_client(Trigger, "fa_cpp/executor_reset")
        assert client.wait_for_service(timeout_sec=2)
        result = client.call_async(Trigger.Request())
        rclpy.spin_until_future_complete(node, result, timeout_sec=2)
        assert result.result().success
        spin(.3)

    try:
        spin(1)
        assert state["commands"] == 0
        spin(.5, send_target=True)
        assert state["commands"] > 100
        spin(.3, send_feedback=False)
        assert "|FAULT" in state["status"] and "|reason=8" in state["status"], state
        frozen = state["commands"]
        spin(.3)
        assert state["commands"] == frozen, "Feedback recovery automatically resumed motion"
        reset()
        spin(.3, send_target=True, wrong_profile=True)
        assert "|reason=1" in state["status"], state
        reset()
        duplicate = node.create_publisher(Float64MultiArray, "upper_position_controller/commands", 1)
        spin(.7)
        assert "|reason=6" in state["status"], state
        node.destroy_publisher(duplicate)
        report = {"passed": True, "feedback_loss_latches": True, "feedback_recovery_does_not_resume": True,
                  "profile_mismatch_rejected": True, "duplicate_output_publisher_rejected": True,
                  "physical_hardware": False, "release_id": identifier.split(":")[0]}
        report["control_backend"] = args.backend
        if args.backend == "mujoco":
            spin(.5)
            reset()
            spin(.3, send_target=True)
            spin(.8, send_target=True, send_backend=False)
            assert "|reason=9" in state["status"], state
            frozen = state["commands"]
            spin(.3)
            assert state["commands"] == frozen, "Backend recovery automatically resumed motion"
            reset()
            spin(.3, wrong_backend=True)
            assert "|reason=9" in state["status"], state
            report.update(backend_loss_latches=True, backend_recovery_does_not_resume=True,
                          backend_version_mismatch_rejected=True)
        (args.output / "result.json").write_text(json.dumps(report, indent=2)+"\n")
        print(json.dumps(report, indent=2))
    finally:
        if proc.poll() is None:
            os.killpg(proc.pid, signal.SIGINT)
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGTERM)
            proc.wait(timeout=5)
        log.close()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
