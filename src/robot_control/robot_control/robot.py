# robot_control/robot_control/robot.py

import numpy as np
import rclpy
from scipy.spatial.transform import Rotation as R
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped

from robot_kinematics_interfaces.srv import GetCurrentPose

class Robot:
    def __init__(self):
        rclpy.init()
        self.tcp_orientation = [np.pi, 0.0, 0.0]
        self.node = Node("robot_control_client")
        self.publisher = self.node.create_publisher(PoseStamped, "/trajectory_goal", 10)

        self.client = self.node.create_client(GetCurrentPose, 'robot_kinematics/get_current_pose')
        if not self.client.wait_for_service(timeout_sec=5.0):
            self.node.get_logger().error("Service 'robot_kinematics/get_current_pose' not available.")

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

    def get_current_pose(self):
        request = GetCurrentPose.Request()
        future = self.client.call_async(request)

        rclpy.spin_until_future_complete(self.node, future)
        response = future.result()
        if response is not None:
            # response should have a 'pose' attribute of type PoseStamped
            pos = response.pose.pose.position
            position = [pos.x, pos.y, pos.z]

            quat = [
                response.pose.pose.orientation.x,
                response.pose.pose.orientation.y,
                response.pose.pose.orientation.z,
                response.pose.pose.orientation.w,
            ]

            rot = R.from_quat(quat)
            tcp_rot = R.from_euler('xyz', self.tcp_orientation)
            tcp_rot_inv = tcp_rot.inv()
            adjusted_rot = rot * tcp_rot_inv
            euler_angles = adjusted_rot.as_euler('xyz')

            return position, euler_angles
        else:
            self.node.get_logger().error("Failed to call service get_current_pose")
            return None, None

    def shutdown(self):
        self.node.destroy_node()
        rclpy.shutdown()
