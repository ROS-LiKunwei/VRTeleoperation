#!/usr/bin/python3
"""Optional recording-only compatibility bridge; never issues robot commands.

Wire format matches the existing fa_left/fa_right BeaVR state dictionaries.
"""
import argparse
import pickle
import time


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--namespace", default="")
    p.add_argument("--left-port", type=int, default=10020)
    p.add_argument("--right-port", type=int, default=10018)
    args = p.parse_args()
    import rclpy
    from rclpy.qos import QoSProfile, ReliabilityPolicy
    from sensor_msgs.msg import JointState
    from std_msgs.msg import Int32
    import zmq

    rclpy.init()
    node = rclpy.create_node("fa_beavr_recording_bridge", namespace=args.namespace)
    context = zmq.Context()
    sockets = []
    for port in [args.left_port, args.right_port]:
        socket = context.socket(zmq.PUB)
        socket.setsockopt(zmq.LINGER, 0)
        socket.setsockopt(zmq.SNDHWM, 1)
        socket.bind(f"tcp://*:{port}")
        sockets.append(socket)
    feedback, commanded = {}, {}
    hand_commands = [1, 1]
    source_stamp = [0.0]
    suffixes = ["shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow", "wrist_yaw", "wrist_pitch", "wrist_roll"]

    def receive(msg):
        stamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        if stamp <= source_stamp[0] or time.time() - stamp > 0.1:
            return
        if len(msg.name) != len(msg.position):
            return
        feedback.clear()
        feedback.update(zip(msg.name, msg.position))
        source_stamp[0] = stamp

    def desired(msg):
        commanded.clear()
        commanded.update(zip(msg.name, msg.position))

    qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT)
    subscriptions = [node.create_subscription(JointState, "joint_states", receive, qos),
                     node.create_subscription(JointState, "fa_cpp/desired_joint_states", desired, qos)]
    for index, side in enumerate(["left", "right"]):
        subscriptions.append(node.create_subscription(Int32, f"{side}_topic_to_hand",
            lambda msg, i=index: hand_commands.__setitem__(i, 2 if msg.data == 20 else 1), 10))

    def publish():
        if time.time() - source_stamp[0] > 0.1:
            return
        for index, side in enumerate(["left", "right"]):
            names = [f"{side}_{name}_joint" for name in suffixes]
            if not all(name in feedback for name in names):
                continue
            positions = [feedback[name] for name in names]
            stamp = source_stamp[0]
            state = {"joint_states": {"joint_position": positions, "timestamp": stamp},
                     "joint_angles_rad": positions, "hand_command": hand_commands[index],
                     "hand_gripper_state": int(hand_commands[index] == 2), "timestamp": stamp}
            if all(name in commanded for name in names):
                state["commanded_joint_states"] = {"joint_position": [commanded[name] for name in names], "timestamp_s": stamp}
            try:
                sockets[index].send_multipart([f"fa_{side}".encode(), pickle.dumps(state, protocol=4)], zmq.NOBLOCK)
            except zmq.Again:
                pass

    timer = node.create_timer(1 / 30, publish)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        for socket in sockets:
            socket.close()
        context.term()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
