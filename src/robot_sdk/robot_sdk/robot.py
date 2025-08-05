# robot_sdk/robot_sdk/robot.py

import numpy as np
import rclpy
from scipy.spatial.transform import Rotation as R
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import JointState
from rclpy.action import ActionClient
from rcl_interfaces.srv import SetParameters, GetParameters
from rcl_interfaces.msg import Parameter, ParameterType
from robot_motion_interfaces.action import CartesianSpaceMotion, JointSpaceMotion
from robot_motion_interfaces.srv import GetCartesianSpacePose, GetJointSpacePose
from robot_motion.types import InterpolationType

class Robot:
    def __init__(self):
        rclpy.init()
        self.node = Node("robot_sdk_client")
        self.tcp_orientation = [np.pi, 0.0, 0.0]

        self.set_param_client = self.node.create_client(
            SetParameters,
            '/robot_motion_node/set_parameters'
        )
        self.get_param_client = self.node.create_client(
            GetParameters,
            '/robot_motion_node/get_parameters'
        )

        self.trajectory_generator = self.TrajectoryGenerator(self)
        self.cartesian_space = self.CartesianSpace(self)
        self.joint_space = self.JointSpace(self)

    def _set_param(self, name: str, value):
        param = Parameter()
        param.name = name

        # Infer type & fill appropriate field
        if isinstance(value, bool):
            param.value.type = ParameterType.PARAMETER_BOOL
            param.value.bool_value = value

        elif isinstance(value, int):
            param.value.type = ParameterType.PARAMETER_INTEGER
            param.value.integer_value = value

        elif isinstance(value, float):
            param.value.type = ParameterType.PARAMETER_DOUBLE
            param.value.double_value = value

        elif isinstance(value, str):
            param.value.type = ParameterType.PARAMETER_STRING
            param.value.string_value = value

        else:
            raise ValueError(f"Unsupported parameter type: {type(value)}")

        req = SetParameters.Request()
        req.parameters = [param]
        future = self.set_param_client.call_async(req)
        rclpy.spin_until_future_complete(self.node, future)
        resp = future.result()
        return resp.results[0].successful

    def _get_param(self, name: str):
        req = GetParameters.Request()
        req.names = [name]
        future = self.get_param_client.call_async(req)
        rclpy.spin_until_future_complete(self.node, future)
        resp = future.result()
        if not resp or not resp.values:
            return None

        val = resp.values[0]
        t = val.type

        if t == ParameterType.PARAMETER_BOOL:
            return val.bool_value

        elif t == ParameterType.PARAMETER_INTEGER:
            return val.integer_value

        elif t == ParameterType.PARAMETER_DOUBLE:
            return val.double_value

        elif t == ParameterType.PARAMETER_STRING:
            return val.string_value

        else:
            # you could handle arrays here (PARAMETER_BYTE_ARRAY, etc.)
            self.node.get_logger().warn(f"Parameter '{name}' has unsupported type {t}")
            return None

    def shutdown(self):
        self.node.destroy_node()
        rclpy.shutdown()

    class TrajectoryGenerator:
        def __init__(self, robot_instance):
            self.robot = robot_instance

        def set_num_waypoints(self, num_waypoints):
            if self.robot._set_param('num_waypoints', num_waypoints):
                self.robot.node.get_logger().info(f"Set num_waypoints to {num_waypoints}")
            else:
                self.robot.node.get_logger().error("Failed to set num_waypoints")

        def get_num_waypoints(self):
            return self.robot._get_param('num_waypoints')
        
        def set_interpolation_type(self, interpolation_type):
            if self.robot._set_param('interpolation_type', interpolation_type):
                self.robot.node.get_logger().info(f"Set interpolation_type to {interpolation_type}")
            else:
                self.robot.node.get_logger().error("Failed to set interpolation_type")
            
        def get_interpolation_type(self):
            return self.robot._get_param('interpolation_type')
        
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
            
        def set_speed(self, speed):
            if self.robot._set_param('joint_space_speed', speed):
                self.robot.node.get_logger().info(f"Set joint space speed to {speed}")
            else:
                self.robot.node.get_logger().error("Failed to set joint space speed")

        def get_speed(self):
            return self.robot._get_param('joint_space_speed')
            
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
            
        def set_speed(self, speed):
            if self.robot._set_param('cartesian_space_speed', speed):
                self.robot.node.get_logger().info(f"Set cartesian space speed to {speed}")
            else:
                self.robot.node.get_logger().error("Failed to set cartesian space speed")

        def get_speed(self):
            return self.robot._get_param('cartesian_space_speed')
