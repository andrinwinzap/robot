# robot_control/robot_control/robot.py

import numpy as np

import rclpy
from scipy.spatial.transform import Rotation as R
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped

class Robot:
    def __init__(self):
        rclpy.init()
        self.tcp_orientation = [np.pi, 0.0, 0.0]
        self.node = Node("robot_control_client")
        self.publisher = self.node.create_publisher(PoseStamped, "/trajectory_goal", 10)

    def move(self, position, orientation=None):
        tcp_rot = R.from_euler('xyz', self.tcp_orientation)

        if orientation is None:
            final_rot = tcp_rot
        else:
            user_rot = R.from_euler('xyz', orientation)
            final_rot = user_rot * tcp_rot

        quat = final_rot.as_quat()

        pose_msg = PoseStamped()
        pose_msg.header.stamp = self.node.get_clock().now().to_msg()
        pose_msg.header.frame_id = "base_link"

        pose_msg.pose.position.x = position[0]
        pose_msg.pose.position.y = position[1]
        pose_msg.pose.position.z = position[2]

        pose_msg.pose.orientation.x = quat[0]
        pose_msg.pose.orientation.y = quat[1]
        pose_msg.pose.orientation.z = quat[2]
        pose_msg.pose.orientation.w = quat[3]

        self.publisher.publish(pose_msg)
        self.node.get_logger().info("Sent desired pose")
    
    def shutdown(self):
        self.node.destroy_node()
        rclpy.shutdown()
