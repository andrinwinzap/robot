import numpy as np

from typing import List, Sequence

from scipy.spatial.transform import Rotation as R, Slerp
from scipy.interpolate import CubicSpline

import rclpy

from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.logging import LoggingSeverity

from std_msgs.msg import Bool, Float32, Float64MultiArray
from control_msgs.action import FollowJointTrajectory
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from controller_manager_msgs.srv import SwitchController

from rcl_interfaces.srv import SetParameters
from rcl_interfaces.msg import Parameter, ParameterType
from builtin_interfaces.msg import Duration

from robot_api.numeric_kinematics import *
import robot_api.config as config


class Tool:
    def __init__(self, robot_instance):
        self.robot = robot_instance
        self._tcp_position = (0.0, 0.0, 0.0)
        self._tcp_orientation = (1.0, 0.0, 0.0, 0.0)


class Robot:
    def __init__(self):
        self._robot_position = config.ROBOT_POSITION
        self._robot_orientation = config.ROBOT_ORIENTATION
        self.joint_velocity_limits = config.JOINT_VELOCITY_LIMITS
        self.joint_acceleration_limits = config.JOINT_ACCELERATION_LIMITS
        self.joint_trajectory_resolution = config.JOINT_TRAJECTORY_RESOLUTION

        self._tcp_position = (0.0, 0.0, 0.0)
        self._tcp_orientation = (1.0, 0.0, 0.0, 0.0)

        self._joint_names = [f"joint_{i+1}" for i in range(6)]
        self._joint_configuration = None
        self._fake_hardware = False

        rclpy.init()

        self.node = Node(
            "robot_api_client", automatically_declare_parameters_from_overrides=True
        )

        self.node.create_subscription(
            JointState, "/joint_states", self._joint_states_callback, 10
        )

        self._trajectory_client = ActionClient(
            self.node,
            FollowJointTrajectory,
            "/joint_trajectory_controller/follow_joint_trajectory",
        )
        self._velocity_controller_command_client = self.node.create_publisher(
            Float64MultiArray, "/velocity_forward_controller/commands", 10
        )

        self._set_hardware_param_client = self.node.create_client(
            SetParameters, "/robot_hardware_interface/set_parameters"
        )

        self._switch_controller_client = self.node.create_client(
            SwitchController, "/controller_manager/switch_controller"
        )

        self.tool_changer = self.ToolChanger(self)
        self.tools = self.Tools(self)
        self.cartesian_space = self.CartesianSpace(self)
        self.joint_space = self.JointSpace(self)

        self.node.get_logger().debug("Waiting for first joint state...")
        while self._joint_configuration is None:
            rclpy.spin_once(self.node, timeout_sec=0.1)
        self.node.get_logger().debug("First joint state received, robot ready.")

    def _switch_controllers(
        self,
        start: list[str],
        stop: list[str],
        strictness: int = SwitchController.Request.BEST_EFFORT,
    ) -> bool:
        if not self._switch_controller_client.wait_for_service(timeout_sec=5.0):
            self.node.get_logger().error("SwitchController service not available")
            return False

        req = SwitchController.Request()
        req.activate_controllers = start
        req.deactivate_controllers = stop
        req.strictness = strictness

        future = self._switch_controller_client.call_async(req)
        rclpy.spin_until_future_complete(self.node, future)
        resp = future.result()

        if resp is None:
            self.node.get_logger().error("SwitchController call failed")
            return False

        self.node.get_logger().info(
            f"Switch controllers: start={start}, stop={stop}, ok={resp.ok}"
        )
        return resp.ok

    def _use_velocity_controller(self) -> bool:
        return self._switch_controllers(
            start=["velocity_forward_controller"], stop=["joint_trajectory_controller"]
        )

    def _use_trajectory_controller(self) -> bool:
        return self._switch_controllers(
            start=["joint_trajectory_controller"], stop=["velocity_forward_controller"]
        )

    def _joint_states_callback(self, msg: JointState):
        joint_map = dict(zip(msg.name, msg.position))
        try:
            self._joint_configuration = [joint_map[name] for name in self._joint_names]
        except KeyError as e:
            self.node.get_logger().warn(f"Missing joint in /joint_states input: {e}")
            return

    def _generate_cubic_spline(self, path: "Robot.JointSpace.Path"):
        points = np.array(path)
        if points.shape[1] != 6:
            raise ValueError("Points must have 6 dimensions (joints).")

        # Extract timestamps from path points
        if any(p.time_from_start is None for p in path.points):
            raise ValueError("All joint-space points must have time_from_start set.")

        timestamps = np.array([p.time_from_start for p in path.points])
        splines = [
            CubicSpline(timestamps, points[:, j], bc_type="clamped") for j in range(6)
        ]

        # Optional: check max velocity / acceleration and rescale if needed
        num_samples = 1000
        t_sample = np.linspace(timestamps[0], timestamps[-1], num_samples)
        max_vel = np.zeros(6)
        max_acc = np.zeros(6)
        for j, s in enumerate(splines):
            vel = s(t_sample, 1)
            acc = s(t_sample, 2)
            max_vel[j] = np.max(np.abs(vel))
            max_acc[j] = np.max(np.abs(acc))

        vel_ratios = max_vel / np.array(self.joint_velocity_limits)
        acc_ratios = max_acc / np.array(self.joint_acceleration_limits)
        scale_factor = max(np.max(vel_ratios), np.sqrt(np.max(acc_ratios)), 1.0)

        if scale_factor > 1.0 + 1e-3:
            # Stretch timestamps to respect limits
            timestamps = timestamps * scale_factor
            splines = [
                CubicSpline(timestamps, points[:, j], bc_type="clamped")
                for j in range(6)
            ]

        return splines, timestamps[-1]

    def _generate_trajectory(self, path: "Robot.JointSpace.Path"):
        trajectory = JointTrajectory()
        trajectory.header.stamp = (
            self.node.get_clock().now() + rclpy.duration.Duration(seconds=0.1)
        ).to_msg()  # small delay
        trajectory.joint_names = self._joint_names

        # Generate cubic splines using the point timestamps
        splines, total_time = self._generate_cubic_spline(path)

        # Use original point timestamps for interpolation
        timestamps = np.array([p.time_from_start for p in path.points])

        # Optionally: generate intermediate points for smooth trajectory
        num_samples = self.joint_trajectory_resolution
        times = np.linspace(timestamps[0], timestamps[-1], num_samples)

        for t in times:
            point = JointTrajectoryPoint()
            point.positions = [float(s(t, 0)) for s in splines]
            point.velocities = [float(s(t, 1)) for s in splines]
            point.accelerations = [float(s(t, 2)) for s in splines]

            dur = Duration()
            dur.sec = int(np.floor(t))
            dur.nanosec = int((t - np.floor(t)) * 1e9)
            point.time_from_start = dur

            trajectory.points.append(point)

        # Ensure last point stops cleanly
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
            self.node.get_logger().debug(
                f"Joint position error: [{formatted_pos_error}]"
            )

            if desired.velocities and actual.velocities:
                vel_error = [
                    d - a for d, a in zip(desired.velocities, actual.velocities)
                ]
                formatted_vel_error = ", ".join(f"{e:+.4f}" for e in vel_error)
                self.node.get_logger().debug(
                    f"Joint velocity error: [{formatted_vel_error}]"
                )

        send_goal_future = self._trajectory_client.send_goal_async(
            fjt_goal, feedback_callback=feedback_callback
        )
        rclpy.spin_until_future_complete(self.node, send_goal_future)
        goal_handle = send_goal_future.result()

        if not goal_handle.accepted:
            self.node.get_logger().error("Trajectory rejected by controller.")
            return False

        self.node.get_logger().debug("Trajectory accepted by controller.")

        # Wait for result while still spinning
        result_future = goal_handle.get_result_async()
        while rclpy.ok() and not result_future.done():
            rclpy.spin_once(self.node, timeout_sec=0.1)

        result = result_future.result().result

        if result.error_code == FollowJointTrajectory.Result.SUCCESSFUL:
            self.node.get_logger().debug("Trajectory completed successfully.")
            return True
        else:
            self.node.get_logger().error(
                f"Controller failed with error code {result.error_code}"
            )
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
            self._command_publisher = self.robot.node.create_publisher(
                Bool, "/robot/tool_changer/attach", 10
            )

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

            self.robot._tcp_position = (0, 0, 0)
            self.robot._tcp_orientation = (0, 0, 0, 0)
            self.current_tool = None

    class Tools:
        def __init__(self, robot_instance):
            self.robot = robot_instance
            self.gripper = self.Gripper(self.robot)

        class Gripper(Tool):
            def __init__(self, robot_instance):
                super().__init__(robot_instance)  # Initialise base attributes
                self._tcp_position = (0.0, 0.0, 0.0618)
                self._tcp_orientation = (1.0, 0.0, 0.0, 0.0)
                self._command_publisher = self.robot.node.create_publisher(
                    Float32, "/robot/gripper/send_command", 10
                )

            def set_distance(self, pos):

                if not self.robot.tool_changer.current_tool == self:
                    raise RuntimeError(f" {self} not the current tool")

                msg = Float32()
                msg.data = float(pos)
                if not self.robot._fake_hardware:
                    self._command_publisher.publish(msg)

    class JointSpace:
        def __init__(self, robot_instance):
            self.robot = robot_instance
            self.speed = 1.0

        def move(self, point: "Robot.JointSpace.Point"):
            if not check_limits(point.joint_configuration):
                self.robot.node.get_logger().error(f"Joint positions not within limits")

            path = Robot.JointSpace.Path()

            start_point = Robot.JointSpace.Point(self.robot._joint_configuration)
            start_point.time_from_start = 0.0
            path.add(start_point)

            dq = np.abs(
                np.array(self.robot._joint_configuration)
                - np.array(point.joint_configuration)
            )
            point.time_from_start = np.max(dq) / self.speed
            path.add(point)

            trajectory = self.robot._generate_trajectory(path)

            return self.robot._send_trajectory(trajectory)

        def set_velocities(self, velocities: Sequence[float]):
            msg = Float64MultiArray()
            msg.data = list(velocities)
            self.robot._velocity_controller_command_client.publish(msg)

        def read(self, decimals: int = 5) -> "Robot.JointSpace.Point":
            if decimals is None:
                joint_configuration = self.robot._joint_configuration
            else:
                joint_configuration = np.round(
                    self.robot._joint_configuration, decimals
                )
            return Robot.JointSpace.Point(joint_configuration)

        class Point:
            def __init__(
                self,
                joint_configuration: Sequence[float] = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
                time_from_start=None,
            ):
                self.joint_configuration = list(joint_configuration)
                self.time_from_start = time_from_start

            def __repr__(self):
                subscripts = "₁₂₃₄₅₆"
                joints = [
                    f"θ{subscripts[i]} = {val: .5f}"
                    for i, val in enumerate(self.joint_configuration)
                ]
                return "Robot.JointSpace.Point(\n  " + "\n  ".join(joints) + "\n)"

            def __array__(self, dtype=None):
                return np.array(self.joint_configuration, dtype=dtype)

        class Path:
            def __init__(self):
                self.points: List["Robot.JointSpace.Point"] = []

            def add(self, point: "Robot.JointSpace.Point"):
                self.points.append(point)

            def __iter__(self):
                return iter(self.points)

            def __len__(self):
                return len(self.points)

            def __array__(self, dtype=None):
                if not self.points:
                    return np.empty((0, 6), dtype=dtype)
                return np.array(
                    [np.array(p, dtype=dtype) for p in self.points], dtype=dtype
                )

    class CartesianSpace:

        def __init__(self, robot_instance):
            self.robot = robot_instance
            self.linear_speed = 0.05
            self.angular_speed = 0.1
            self.linear_acceleration = 0.05
            self.interpolation_step_size = 0.01

        def _robot_to_tcp(self):
            pos = np.array(self.robot._tcp_position, float)
            quat = np.array(self.robot._tcp_orientation, float)

            if np.linalg.norm(quat) == 0:
                self.robot.node.get_logger().warn(
                    "_tcp_orientation quaternion must not be zero."
                )
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
            T[:3, :3] = R.from_quat(quat).as_matrix()
            T[:3, 3] = pos
            return T

        def _base_to_world(self):
            return np.linalg.inv(self._world_to_base())

        def _interpolate_htm(self, start_T, end_T, t):
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

        def _trapezoidal_profile(self, distance, v_max, a_max, dt=0.001):
            s = []
            times = []
            t = 0.0
            pos = vel = 0.0

            t_acc = v_max / a_max
            d_acc = 0.5 * a_max * t_acc**2

            if 2 * d_acc > distance:
                t_acc = np.sqrt(distance / a_max)
                t_total = 2 * t_acc
            else:
                d_cruise = distance - 2 * d_acc
                t_cruise = d_cruise / v_max
                t_total = 2 * t_acc + t_cruise

            while t < t_total + 1e-6:
                if t < t_acc:
                    pos = 0.5 * a_max * t**2
                elif t < t_total - t_acc:
                    pos = d_acc + v_max * (t - t_acc)
                else:
                    dt_dec = t - (t_total - t_acc)
                    pos = distance - 0.5 * a_max * (t_acc - dt_dec) ** 2
                s.append(pos)
                times.append(t)
                t += dt
            return s, times

        def move(
            self, pose: "Robot.CartesianSpace.Pose", enforce_linearity: bool = True
        ):
            start = (
                self._world_to_base()
                @ forward_kinematics(self.robot._joint_configuration)
                @ self._robot_to_tcp()
            )
            end = pose.as_matrix()

            linear_dist = np.linalg.norm(start[:3, 3] - end[:3, 3])

            R_start = start[:3, :3]
            R_end = end[:3, :3]
            R_diff = R_end @ R_start.T
            rot_vec = R.from_matrix(R_diff).as_rotvec()
            angular_dist = np.linalg.norm(rot_vec)

            weight_factor = self.linear_speed / self.angular_speed
            effective_distance = max(linear_dist, angular_dist * weight_factor)

            path = Robot.CartesianSpace.Path()

            if enforce_linearity:
                s_profile, t_profile = self._trapezoidal_profile(
                    effective_distance,
                    v_max=self.linear_speed,
                    a_max=self.linear_acceleration,
                    dt=self.interpolation_step_size,
                )

                for s, t in zip(s_profile, t_profile):
                    alpha = s / effective_distance if effective_distance > 1e-9 else 1.0
                    T = self._interpolate_htm(start, end, alpha)
                    pose = Robot.CartesianSpace.Pose.from_matrix(T)
                    pose.time_from_start = t
                    path.add(pose)

            else:
                start_pose = Robot.CartesianSpace.Pose.from_matrix(start)
                start_pose.time_from_start = 0.0

                end_pose = pose

                time_linear = linear_dist / self.linear_speed
                time_angular = angular_dist / self.angular_speed

                total_duration = max(time_linear, time_angular)

                end_pose.time_from_start = total_duration

                path.add(start_pose)
                path.add(end_pose)

            return self.follow_path(path)

        def follow_path(self, path: "Robot.CartesianSpace.Path"):

            joint_space_path = Robot.JointSpace.Path()

            prev_joint_configuration = self.robot._joint_configuration
            for i, pose in enumerate(path):
                T = self._base_to_world() @ pose.as_matrix() @ self._tcp_to_robot()

                ik_solutions = inverse_kinematics(T)
                if not ik_solutions:
                    self.robot.node.get_logger().error(
                        f"No IK solution found at pose {i}"
                    )
                    return False
                prev_joint_configuration = chose_optimal_solution(
                    prev_joint_configuration, ik_solutions
                )
                point = Robot.JointSpace.Point(prev_joint_configuration)
                point.time_from_start = pose.time_from_start
                joint_space_path.add(point)

            offset = np.linalg.norm(
                np.array(joint_space_path.points[0]) - self.robot._joint_configuration
            )

            if offset > 1e-1:
                self.robot.node.get_logger().error("Robot not at start of path")
                return False

            trajectory = self.robot._generate_trajectory(joint_space_path)

            return self.robot._send_trajectory(trajectory)

        def read(self):
            T = (
                self._world_to_base()
                @ forward_kinematics(self.robot._joint_configuration)
                @ self._robot_to_tcp()
            )
            position = T[:3, 3]
            orientation = R.from_matrix(T[:3, :3]).as_euler("xyz")
            pose = Robot.CartesianSpace.Pose(position, orientation)
            return pose

        class Pose:
            def __init__(
                self,
                position: Sequence[float] = (0.0, 0.0, 0.0),
                orientation: Sequence[float] = (0.0, 0.0, 0.0),
                time_from_start=None,
            ):
                self.position = list(position)
                self.orientation = list(orientation)
                self.time_from_start = time_from_start

            def as_matrix(self):
                T = np.eye(4)
                T[:3, 3] = np.array(self.position)
                T[:3, :3] = R.from_euler("xyz", self.orientation).as_matrix()
                return T

            @classmethod
            def from_matrix(cls, T: np.ndarray) -> "Robot.CartesianSpace.Pose":
                if T.shape != (4, 4):
                    raise ValueError("Input must be a 4x4 homogeneous transform")
                position = T[:3, 3]
                orientation = R.from_matrix(T[:3, :3]).as_euler("xyz")
                return cls(position, orientation)

            def __repr__(self):
                position = np.round(self.position, 5)
                orientation = np.round(self.orientation, 5)
                return (
                    f"Robot.CartesianSpace.Pose(\n"
                    f"  Position:    X={position[0]}, Y={position[1]}, Z={position[2]}\n"
                    f"  Orientation: Roll={orientation[0]}, Pitch={orientation[1]}, Yaw={orientation[2]}\n"
                    f")"
                )

        class Path:
            def __init__(self):
                self.poses: List["Robot.CartesianSpace.Pose"] = []

            def add(self, pose: "Robot.CartesianSpace.Pose"):
                self.poses.append(pose)

            def length(self):
                return sum(
                    np.linalg.norm(np.array(p2.position) - np.array(p1.position))
                    for p1, p2 in zip(self.poses[:-1], self.poses[1:])
                )

            def __iter__(self):
                return iter(self.poses)

            def __len__(self):
                return len(self.poses)
