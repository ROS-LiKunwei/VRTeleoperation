#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray

class EffortPublisher(Node):
    def __init__(self):
        super().__init__('effort_publisher')
        self.publisher_ = self.create_publisher(Float64MultiArray, '/effort_controller/commands', 1000)
        self.timer = self.create_timer(0.001, self.timer_callback)  # 0.001s = 1000Hz

    def timer_callback(self):
        msg = Float64MultiArray()
        msg.data = [1.0, 2.0,3.0,4.0,5.0,6.0,7.0,8.0,9.0,10.0,11.0,12.0,13.0,14.0,15.0,16.0,17.0,18.0,19.0,20.0,21.0,22.0,23.0,24.0,25.0,26.0]
        self.publisher_.publish(msg)
        self.get_logger().info(f'Publishing: "{msg.data}"')

def main(args=None):
    rclpy.init(args=args)
    node = EffortPublisher()
    rclpy.spin(node)

    # Shutdown
    rclpy.shutdown()

if __name__ == '__main__':
    main()
