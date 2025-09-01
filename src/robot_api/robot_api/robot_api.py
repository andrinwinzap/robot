# robot_api/robot_api/robot.py

import numpy as np
import time

from scipy.spatial.transform import Rotation as R, Slerp
from scipy.interpolate import CubicSpline

from robot_api.numeric_kinematics import forward_kinematics, inverse_kinematics, check_limits, chose_optimal_solution

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient, ActionServer
from rclpy.logging import LoggingSeverity
from std_msgs.msg import Bool, Float32
from control_msgs.action import FollowJointTrajectory
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from rcl_interfaces.srv import SetParameters, GetParameters
from rcl_interfaces.msg import Parameter, ParameterType
from builtin_interfaces.msg import Duration
from rclpy.logging import LoggingSeverity

class Tool:
    def __init__(self, robot_instance):
        self.robot = robot_instance
        self._tcp_position = (0.0,0.0,0.0)
        self._tcp_orientation = (0.0, 0.0, 0.0)

class Robot:
    def __init__(self):
        self._joint_names = [f"joint_{i+1}" for i in range(6)]
        self._joint_configuration = None
        self._tcp_position = [0.0, 0.0, 0.0]
        self._tcp_orientation = [1.0, 0.0, 0.0, 0.0]
        self._robot_position = [0.0, 0.35, 0.0]
        self._robot_orientation = [0.0, 0.0, 0.0, 1.0]
        self._fake_hardware = False

        self.trajectory_resolution = 50
        self.joint_velocity_limits = [5.0, 5.0, 5.0, 5.0, 5.0, 5.0]
        self.joint_acceleration_limits = [3.14, 3.14, 3.14, 3.14, 3.14, 3.14]
        
        rclpy.init()

        self.node = Node(
                "robot_api_client",
                automatically_declare_parameters_from_overrides=True
            )

        self.node.create_subscription(JointState, '/joint_states', self._joint_states_callback, 10)

        self._trajectory_client = ActionClient(self.node, FollowJointTrajectory, '/joint_trajectory_controller/follow_joint_trajectory')

        self._set_hardware_param_client = self.node.create_client(
            SetParameters,
            '/robot_hardware_interface/set_parameters'
        )

        self.tool_changer = self.ToolChanger(self)
        self.tools = self.Tools(self)
        self.cartesian_space = self.CartesianSpace(self)
        self.joint_space = self.JointSpace(self)

        self.node.get_logger().info("Waiting for first joint state...")
        while self._joint_configuration is None:
            rclpy.spin_once(self.node, timeout_sec=0.1)
        self.node.get_logger().info("First joint state received, robot ready.")

    def _joint_states_callback(self, msg: JointState):
        joint_map = dict(zip(msg.name, msg.position))
        try:
            self._joint_configuration = [joint_map[name] for name in self._joint_names]
        except KeyError as e:
            self.node.get_logger().warn(f"Missing joint in /joint_states input: {e}")
            return

    def _generate_cubic_spline(self, points, proposed_time):
        points = np.array(points)
        if points.shape[1] != 6:
            raise ValueError("Points must have 6 dimensions (joints).")
        
        actual_time = proposed_time
        num_samples = 1000
        while True:
            num_points = points.shape[0]
            timestamps = np.linspace(0, actual_time, num_points)
            splines = [CubicSpline(timestamps, points[:, j], bc_type='clamped') for j in range(6)]
            
            t_sample = np.linspace(0, actual_time, num_samples)
            max_joint_vel = 0
            max_joint_acc = 0
            for s in splines:
                vel = s(t_sample, 1)
                acc = s(t_sample, 2)
                max_joint_vel = max(max_joint_vel, np.max(np.abs(vel)))
                max_joint_acc = max(max_joint_acc, np.max(np.abs(acc)))
            
            vel_ratio = max_joint_vel / max(self.joint_velocity_limits)
            acc_ratio = max_joint_acc / max(self.joint_acceleration_limits)
            scale_factor = max(vel_ratio, np.sqrt(acc_ratio), 1.0)  # at least 1
            
            if scale_factor <= 1.0:
                break
            actual_time *= scale_factor
        
        return splines, actual_time

    def _generate_trajectory(self, points, proposed_time):
        trajectory = JointTrajectory()
        trajectory.header.stamp = (self.node.get_clock().now() +
                                rclpy.duration.Duration(seconds=0.1)).to_msg()  # small delay
        trajectory._joint_names = self._joint_names

        splines, actual_time = self._generate_cubic_spline(points, proposed_time)

        num_points = self.trajectory_resolution
        times = np.linspace(0, actual_time, num_points)

        for t in times:
            point = JointTrajectoryPoint()
            point.positions = [s(t, 0) for s in splines]
            point.velocities = [s(t, 1) for s in splines]
            point.accelerations = [s(t, 2) for s in splines]

            dur = Duration()
            dur.sec = int(np.floor(t))
            dur.nanosec = int((t - np.floor(t)) * 1e9)
            point.time_from_start = dur

            trajectory.points.append(point)

        trajectory.points[-1].velocities = [0.0] * len(self._joint_names)
        trajectory.points[-1].accelerations = [0.0] * len(self._joint_names)

        return trajectory

    def _send_trajectory(self, trajectory):
        if not self._trajectory_client.wait_for_server(timeout_sec=5.0):
            self.node.get_logger().error("FollowJointTrajectory server not available.")
            return False

        fjt_goal = FollowJointTrajectory.Goal()
        fjt_goal.trajectory = trajectory

        def feedback_callback(feedback_msg):
            feedback = feedback_msg.feedback
            desired = feedback.desired
            actual = feedback.actual

            pos_error = [d - a for d, a in zip(desired.positions, actual.positions)]
            formatted_pos_error = ", ".join(f"{e:+.4f}" for e in pos_error)
            self.node.get_logger().debug(f"Joint position error: [{formatted_pos_error}]")

            if desired.velocities and actual.velocities:
                vel_error = [d - a for d, a in zip(desired.velocities, actual.velocities)]
                formatted_vel_error = ", ".join(f"{e:+.4f}" for e in vel_error)
                self.node.get_logger().debug(f"Joint velocity error: [{formatted_vel_error}]")


        send_goal_future = self._trajectory_client.send_goal_async(fjt_goal, feedback_callback=feedback_callback)
        rclpy.spin_until_future_complete(self.node, send_goal_future)
        goal_handle = send_goal_future.result()

        if not goal_handle.accepted:
            self.node.get_logger().error("Trajectory rejected by controller.")
            return False

        self.node.get_logger().info("Trajectory accepted by controller.")

        # Wait for result while still spinning
        result_future = goal_handle.get_result_async()
        while rclpy.ok() and not result_future.done():
            rclpy.spin_once(self.node, timeout_sec=0.1)

        result = result_future.result().result

        if result.error_code == FollowJointTrajectory.Result.SUCCESSFUL:
            self.node.get_logger().info("Trajectory completed successfully.")
            return True
        else:
            self.node.get_logger().error(f"Controller failed with error code {result.error_code}")
            return False
        
    def set_fake_hardware(self, value):
        param = Parameter()
        param.name = "fake_hardware"
        param.value.type = ParameterType.PARAMETER_BOOL
        param.value.bool_value = value
        req = SetParameters.Request()
        req.parameters = [param]
        future = self._set_hardware_param_client.call_async(req)
        rclpy.spin_until_future_complete(self.node, future)
        resp = future.result()
        if not resp.results[0].successful:
            raise RuntimeError("Failed to set simulation mode")
        self._fake_hardware = value
    
    def set_debug_mode(self, value):
        if value:
            self.node.get_logger().set_level(LoggingSeverity.DEBUG)
        else:
            self.node.get_logger().set_level(LoggingSeverity.INFO)
            
    def shutdown(self):
        self.node.destroy_node()
        rclpy.shutdown()

    class ToolChanger:
        def __init__(self, robot_instance):
            self.robot = robot_instance
            self.current_tool = None
            self._command_publisher = self.robot.node.create_publisher(Bool, '/robot/tool_changer/attach', 10)
        
        def attach_tool(self, tool: Tool):
            msg = Bool()
            msg.data = True
            self._command_publisher.publish(msg)

            self.robot._tcp_position = tool._tcp_position
            self.robot._tcp_orientation = tool._tcp_orientation

            self.current_tool = tool

        def detach_tool(self):
            msg = Bool()
            msg.data = False

            if not self.robot._fake_hardware:
                self._command_publisher.publish(msg)

            self.robot._tcp_position = (0,0,0)
            self.robot._tcp_position = (0,0,0,0)
            self.current_tool = None

    class Tools:
        def __init__(self, robot_instance):
            self.robot = robot_instance
            self.gripper = self.Gripper(self.robot)

        class Gripper(Tool):
            def __init__(self, robot_instance):
                super().__init__(robot_instance)  # Initialise base attributes
                self._tcp_position = (0.0, 0.0, 0.059)
                self._tcp_orientation = (0.0, 0.0, 0.0)
                self._command_publisher = self.robot.node.create_publisher(
                    Float32, '/robot/gripper/send_command', 10
                )

            def set_distance(self, pos):
                
                if not self.robot.tool_changer.current_tool==self:
                    raise RuntimeError(f" {self} not the current tool")
                
                msg = Float32()
                msg.data = float(pos)
                if not self.robot._fake_hardware:
                    self._command_publisher.publish(msg)
        
    class JointSpace:
        def __init__(self, robot_instance):
            self.robot = robot_instance
            self.speed = 1.0

        def move(self, joint_positions):
            if not check_limits(joint_positions):
                self.robot.node.get_logger().error(f"Joint positions not within limits")

            points = np.array([np.array(self.robot._joint_configuration), np.array(joint_positions)])

            dq = np.abs(np.array(self.robot._joint_configuration) - np.array(joint_positions))
            proposed_time = np.max(dq) / self.speed

            trajectory = self.robot._generate_trajectory(points, proposed_time)

            return self.robot._send_trajectory(trajectory)

        def get_pose(self):
            return self.robot._joint_configuration
            
    class CartesianSpace:

        def __init__(self, robot_instance):
            self.robot = robot_instance
            self.speed = 0.05
        
        def _robot_to_tcp(self):
            pos = np.array(self.robot._tcp_position, float)
            quat = np.array(self.robot._tcp_orientation, float)

            if np.linalg.norm(quat) == 0:
                self.robot.node.get_logger().warn("_tcp_orientation quaternion must not be zero.")
                return np.eye(4)

            quat /= np.linalg.norm(quat)

            T = np.eye(4)
            T[:3, :3] = R.from_quat(quat).as_matrix()
            T[:3, 3] = pos
            return T
        
        def _tcp_to_robot(self):
            return np.linalg.inv(self._robot_to_tcp())

        def _world_to_base(self):
            pos = self.robot._robot_position
            quat = self.robot._robot_orientation
            quat /= np.linalg.norm(quat)
            T = np.eye(4)
            T[:3,:3] = R.from_quat(quat).as_matrix()
            T[:3, 3] = pos
            return T
        
        def _base_to_world(self):
            return np.linalg.inv(self._world_to_base())

        def _interpolate_pose(self, start_T, end_T, t):
            start_p = start_T[:3, 3]
            end_p = end_T[:3, 3]
            interp_p = start_p * (1 - t) + end_p * t

            rotations = R.from_matrix([start_T[:3, :3], end_T[:3, :3]])
            slerp = Slerp([0, 1], rotations)
            interp_R = slerp([t])[0].as_matrix()

            T = np.eye(4)
            T[:3, :3] = interp_R
            T[:3, 3] = interp_p
            return T
    
        def move(self, position, orientation=None):
            if orientation is None:
                orientation = [0.0, 0.0, 0.0]

            tcp_world_T = np.eye(4)
            tcp_world_T[:3, 3] = np.array(position)
            tcp_world_T[:3, :3] = R.from_euler('xyz', orientation).as_matrix()

            end_T = self._base_to_world() @ tcp_world_T @ self._tcp_to_robot()

            start_T = forward_kinematics(self.robot._joint_configuration)

            points=[]
            prev_joints = self.robot._joint_configuration
            for i in range(self.robot.trajectory_resolution):
                alpha = i/(self.robot.trajectory_resolution - 1)
                T = self._interpolate_pose(start_T, end_T, alpha)
                ik_solutions = inverse_kinematics(T)
                if not ik_solutions:
                    self.robot.node.get_logger().error(f"No IK solution found at alpha={alpha}")
                    return False
                prev_joints = chose_optimal_solution(prev_joints, ik_solutions)
                points.append(prev_joints)

            start_pos = np.array(start_T[:3, 3])
            end_pos = np.array(end_T[:3, 3])
            dist = np.linalg.norm(end_pos - start_pos)

            time_cartesian_space = dist / self.speed

            trajectory = self.robot._generate_trajectory(points, time_cartesian_space)

            return self.robot._send_trajectory(trajectory)

        def get_pose(self):
            T = self._world_to_base() @ forward_kinematics(self.robot._joint_configuration) @ self._robot_to_tcp()
            position = T[:3, 3]
            orientation = R.from_matrix(T[:3, :3]).as_euler("xyz")
            return position, orientation
