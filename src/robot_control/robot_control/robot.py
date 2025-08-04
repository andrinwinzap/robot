# robot_control/robot_control/robot.py

import numpy as np
import rclpy
from scipy.spatial.transform import Rotation as R
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import JointState

from robot_kinematics_interfaces.srv import GetCurrentPose, GetJointConfiguration

class Robot:
    def __init__(self):
        rclpy.init()
        self.tcp_orientation = [np.pi, 0.0, 0.0]
        self.node = Node("robot_control_client")
        self.publisher = self.node.create_publisher(PoseStamped, "/trajectory_goal", 10)
        self.joint_goal_pub = self.node.create_publisher(JointState, '/joint_goal', 10)

        self.pose_client = self.node.create_client(GetCurrentPose, 'robot_kinematics/get_current_pose')
        self.joint_config_client = self.node.create_client(GetJointConfiguration, 'robot_kinematics/get_joint_configuration')

        if not self.pose_client.wait_for_service(timeout_sec=5.0):
            self.node.get_logger().error("Service 'robot_kinematics/get_current_pose' not available.")
        if not self.joint_config_client.wait_for_service(timeout_sec=5.0):
            self.node.get_logger().error("Service 'robot_kinematics/get_joint_configuration' not available.")

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
        future = self.pose_client.call_async(request)

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

    def get_joint_configuration(self):
        request = GetJointConfiguration.Request()
        future = self.joint_config_client.call_async(request)
        rclpy.spin_until_future_complete(self.node, future)
        response = future.result()
        if response is not None:
            return dict(zip(response.joint_names, response.joint_positions))
        else:
            self.node.get_logger().error("Failed to call service get_joint_configuration")
            return {}
        
    def move_joint_space(self, joint_positions):
        msg = JointState()
        msg.header.stamp = self.node.get_clock().now().to_msg()
        msg.name = [f"joint_{i+1}" for i in range(len(joint_positions))]
        msg.position = joint_positions
        self.joint_goal_pub.publish(msg)
        self.node.get_logger().info(f"Published joint_goal with positions: {joint_positions}")

    def shutdown(self):
        self.node.destroy_node()
        rclpy.shutdown()
