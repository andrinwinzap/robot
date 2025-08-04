# robot_sdk/robot_sdk/robot.py

import numpy as np
import rclpy
from scipy.spatial.transform import Rotation as R
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import JointState
from robot_motion_interfaces.srv import GetCartesianSpacePose, GetJointSpacePose

class Robot:
    def __init__(self):
        rclpy.init()
        self.node = Node("robot_sdk_client")
        self.tcp_orientation = [np.pi, 0.0, 0.0]

        self.cartesian_space = self.CartesianSpace(self)
        self.joint_space = self.JointSpace(self)
        
    def shutdown(self):
        self.node.destroy_node()
        rclpy.shutdown()

    class JointSpace:
        def __init__(self, robot_instance):
            self.robot = robot_instance

            self.pose_setter_publisher = self.robot.node.create_publisher(JointState, '/robot_motion/joint_space/set_goal_pose', 10)
            self.pose_getter_client = self.robot.node.create_client(GetJointSpacePose, '/robot_motion/joint_space/get_pose')
            if not self.pose_getter_client.wait_for_service(timeout_sec=5.0):
                self.robot.node.get_logger().error("Service '/robot_motion/joint_space/get_pose' not available.")
                
        def move(self, joint_positions):
            msg = JointState()
            msg.header.stamp = self.robot.node.get_clock().now().to_msg()
            msg.name = [f"joint_{i+1}" for i in range(len(joint_positions))]
            msg.position = joint_positions
            self.pose_setter_publisher.publish(msg)
            self.robot.node.get_logger().info(f"Published joint_goal with positions: {joint_positions}")

        def get_pose(self):
            request = GetJointSpacePose.Request()
            future = self.pose_getter_client.call_async(request)
            rclpy.spin_until_future_complete(self.robot.node, future)
            response = future.result()
            if response is not None:
                return dict(zip(response.joint_names, response.joint_positions))
            else:
                self.robot.node.get_logger().error("Failed to call service get_joint_configuration")
                return {}
            
    class CartesianSpace:

        def __init__(self, robot_instance):
            self.robot = robot_instance

            self.pose_setter_publisher = self.robot.node.create_publisher(PoseStamped, "/robot_motion/cartesian_space/set_goal_pose", 10)
            self.pose_getter_client = self.robot.node.create_client(GetCartesianSpacePose, '/robot_motion/cartesian_space/get_pose')
            if not self.pose_getter_client.wait_for_service(timeout_sec=5.0):
                self.robot.node.get_logger().error("Service '/robot_motion/cartesian_space/get_pose' not available.")

        def move(self, position, orientation=None):
            tcp_rot = R.from_euler('xyz', self.robot.tcp_orientation)

            if orientation is None:
                final_rot = tcp_rot
            else:
                user_rot = R.from_euler('xyz', orientation)
                final_rot = user_rot * tcp_rot

            quat = final_rot.as_quat()

            pose_msg = PoseStamped()
            pose_msg.header.stamp = self.robot.node.get_clock().now().to_msg()
            pose_msg.header.frame_id = "base_link"
            pose_msg.pose.position.x = position[0]
            pose_msg.pose.position.y = position[1]
            pose_msg.pose.position.z = position[2]
            pose_msg.pose.orientation.x = quat[0]
            pose_msg.pose.orientation.y = quat[1]
            pose_msg.pose.orientation.z = quat[2]
            pose_msg.pose.orientation.w = quat[3]

            self.pose_setter_publisher.publish(pose_msg)
            self.robot.node.get_logger().info("Sent desired pose")

        def get_pose(self):
            request = GetCartesianSpacePose.Request()
            future = self.pose_getter_client.call_async(request)

            rclpy.spin_until_future_complete(self.robot.node, future)
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
                tcp_rot = R.from_euler('xyz', self.robot.tcp_orientation)
                tcp_rot_inv = tcp_rot.inv()
                adjusted_rot = rot * tcp_rot_inv
                euler_angles = adjusted_rot.as_euler('xyz')

                return position, euler_angles
            else:
                self.robot.node.get_logger().error("Failed to call service get_current_pose")
                return None, None
