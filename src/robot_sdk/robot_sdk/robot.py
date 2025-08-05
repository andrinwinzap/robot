# robot_sdk/robot_sdk/robot.py

import numpy as np
import rclpy
from scipy.spatial.transform import Rotation as R
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import JointState
from rclpy.action import ActionClient
from robot_motion_interfaces.action import CartesianSpaceMotion, JointSpaceMotion
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

            self.pose_setter_client = ActionClient(
                self.robot.node,
                JointSpaceMotion,
                '/robot_motion/joint_space/motion')

            self.pose_getter_client = self.robot.node.create_client(GetJointSpacePose, '/robot_motion/joint_space/get_pose')
            if not self.pose_getter_client.wait_for_service(timeout_sec=5.0):
                self.robot.node.get_logger().error("Service '/robot_motion/joint_space/get_pose' not available.")

        def move(self, joint_positions):
            goal_msg = JointSpaceMotion.Goal()
            goal_msg.joint_state = JointState()
            goal_msg.joint_state.header.stamp = self.robot.node.get_clock().now().to_msg()
            goal_msg.joint_state.name = [f"joint_{i+1}" for i in range(len(joint_positions))]
            goal_msg.joint_state.position = joint_positions

            self.pose_setter_client.wait_for_server()

            def feedback_callback(feedback_msg):
                feedback = feedback_msg.feedback

                # Extract arrays from feedback
                desired_positions = getattr(feedback, 'desired_positions', [])
                desired_velocities = getattr(feedback, 'desired_velocities', [])
                actual_positions = getattr(feedback, 'actual_positions', [])
                actual_velocities = getattr(feedback, 'actual_velocities', [])

                # Format the info string to log
                info_msg = (
                    f"JointSpaceMotion feedback:\n"
                    f"  Desired Positions: {desired_positions}\n"
                    f"  Desired Velocities: {desired_velocities}\n"
                    f"  Actual Positions: {actual_positions}\n"
                    f"  Actual Velocities: {actual_velocities}"
                )
                self.robot.node.get_logger().debug(info_msg)
                
            goal_future = self.pose_setter_client.send_goal_async(goal_msg, feedback_callback=feedback_callback)
            rclpy.spin_until_future_complete(self.robot.node, goal_future)
            goal_handle = goal_future.result()
            if not goal_handle.accepted:
                self.robot.node.get_logger().error("Goal rejected by the action server.")
                return

            result_future = goal_handle.get_result_async()
            rclpy.spin_until_future_complete(self.robot.node, result_future)
            result = result_future.result().result

            self.robot.node.get_logger().info(f"Action result: success={result.success}, message='{result.message}'")

        def get_pose(self):
            request = GetJointSpacePose.Request()
            future = self.pose_getter_client.call_async(request)
            rclpy.spin_until_future_complete(self.robot.node, future)
            response = future.result()
            
            if response is not None:
                return np.array(response.joint_positions, dtype=np.float64)
            else:
                self.robot.node.get_logger().error("Failed to call service get_joint_configuration")
                return None
            
    class CartesianSpace:

        def __init__(self, robot_instance):
            self.robot = robot_instance

            self.pose_setter_client = ActionClient(
                self.robot.node,
                CartesianSpaceMotion,
                '/robot_motion/cartesian_space/motion')

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

            goal_msg = CartesianSpaceMotion.Goal()
            goal_msg.pose = pose_msg

            self.pose_setter_client.wait_for_server()

            def feedback_callback(feedback_msg):
                feedback = feedback_msg.feedback

                # Extract arrays from feedback
                desired_positions = getattr(feedback, 'desired_positions', [])
                desired_velocities = getattr(feedback, 'desired_velocities', [])
                actual_positions = getattr(feedback, 'actual_positions', [])
                actual_velocities = getattr(feedback, 'actual_velocities', [])

                # Format the info string to log
                info_msg = (
                    f"CartesianSpaceMotion feedback:\n"
                    f"  Desired Positions: {desired_positions}\n"
                    f"  Desired Velocities: {desired_velocities}\n"
                    f"  Actual Positions: {actual_positions}\n"
                    f"  Actual Velocities: {actual_velocities}"
                )
                self.robot.node.get_logger().debug(info_msg)

            goal_future = self.pose_setter_client.send_goal_async(goal_msg, feedback_callback=feedback_callback)
            rclpy.spin_until_future_complete(self.robot.node, goal_future)
            goal_handle = goal_future.result()
            if not goal_handle.accepted:
                self.robot.node.get_logger().error("Goal rejected by the action server.")
                return

            result_future = goal_handle.get_result_async()
            rclpy.spin_until_future_complete(self.robot.node, result_future)
            result = result_future.result().result

            self.robot.node.get_logger().info(f"Action result: success={result.success}, message='{result.message}'")

        def get_pose(self):
            request = GetCartesianSpacePose.Request()
            future = self.pose_getter_client.call_async(request)

            rclpy.spin_until_future_complete(self.robot.node, future)
            response = future.result()
            if response is not None:
                pos = response.pose.pose.position
                position = np.array([pos.x, pos.y, pos.z])

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
