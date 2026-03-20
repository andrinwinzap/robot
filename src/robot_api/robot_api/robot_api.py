import atexit
import signal
import time
from threading import Thread
from typing import Sequence

import numpy as np
from scipy.interpolate import CubicSpline

import rclpy
from rclpy.action import ActionClient
from rclpy.executors import SingleThreadedExecutor
from rclpy.logging import LoggingSeverity
from rclpy.node import Node

from builtin_interfaces.msg import Duration
from control_msgs.action import FollowJointTrajectory
from controller_manager_msgs.srv import SwitchController
from rcl_interfaces.msg import Parameter, ParameterType
from rcl_interfaces.srv import SetParameters
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

import robot_api.config as config
from robot_api.joint_space import JointSpace
from robot_api.cartesian_space import CartesianSpace
from robot_api.tools import ToolChanger, Tools


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
        self._controller_type = None
        self._fake_hardware = False

        rclpy.init()
        signal.signal(signal.SIGINT, lambda *_: exit(0))

        self.node = Node(
            "robot_api_client", automatically_declare_parameters_from_overrides=True
        )

        self.node.create_subscription(
            JointState, "/joint_states", self._joint_states_callback, 10
        )

        self._trajectory_controller_client = ActionClient(
            self.node,
            FollowJointTrajectory,
            "/joint_trajectory_controller/follow_joint_trajectory",
        )
        self._velocity_controller_client = self.node.create_publisher(
            Float64MultiArray, "/joint_velocity_controller/commands", 10
        )

        self._set_hardware_param_client = self.node.create_client(
            SetParameters, "/robot_hardware_interface/set_parameters"
        )

        self._switch_controller_client = self.node.create_client(
            SwitchController, "/controller_manager/switch_controller"
        )

        self.tool_changer = ToolChanger(self)
        self.tools = Tools(self)
        self.cartesian_space = CartesianSpace(self)
        self.joint_space = JointSpace(self)

        self._executor = SingleThreadedExecutor()
        self._executor.add_node(self.node)
        self._executor_thread = Thread(target=self._executor.spin, daemon=True)
        self._executor_thread.start()

        self.node.get_logger().debug("Waiting for first joint state...")
        while self._joint_configuration is None:
            time.sleep(0.001)
        self.node.get_logger().debug("First joint state received, robot ready.")
        atexit.register(self.shutdown)

    def set_fake_hardware_mode(self, value):
        param = Parameter()
        param.name = "fake_hardware"
        param.value.type = ParameterType.PARAMETER_BOOL
        param.value.bool_value = value
        req = SetParameters.Request()
        req.parameters = [param]
        future = self._set_hardware_param_client.call_async(req)
        resp = self._wait_for_future(future).result()
        if not resp.results[0].successful:
            raise RuntimeError("Failed to set simulation mode")
        self._fake_hardware = value

    def set_debug_mode(self, value):
        if value:
            self.node.get_logger().set_level(LoggingSeverity.DEBUG)
        else:
            self.node.get_logger().set_level(LoggingSeverity.INFO)

    def shutdown(self):
        if not rclpy.ok():
            return
        self.cartesian_space._stop_twist_timer()
        # Deactivate all controllers so the hardware returns to idle mode
        active = [c for c in ["joint_trajectory_controller", "joint_velocity_controller"]
                  if self._controller_type == c]
        if active:
            self._switch_controllers(start=[], stop=active)
            self._controller_type = None
        self._executor.shutdown()
        self._executor_thread.join()
        self.node.destroy_node()
        rclpy.shutdown()

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
        resp = self._wait_for_future(future).result()

        if resp is None:
            self.node.get_logger().error("SwitchController call failed")
            return False

        self.node.get_logger().info(
            f"Switch controllers: start={start}, stop={stop}, ok={resp.ok}"
        )
        return resp.ok

    def _use_velocity_controller(self) -> bool:
        if self._controller_type == "joint_velocity_controller":
            return True
        if self._switch_controllers(
            start=["joint_velocity_controller"], stop=["joint_trajectory_controller"]
        ):
            self._controller_type = "joint_velocity_controller"
            return True
        return False

    def _use_trajectory_controller(self) -> bool:
        if self._controller_type == "joint_trajectory_controller":
            return True
        self.cartesian_space._stop_twist_timer()
        if self._switch_controllers(
            start=["joint_trajectory_controller"], stop=["joint_velocity_controller"]
        ):
            self._controller_type = "joint_trajectory_controller"
            return True
        return False

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
        self._use_trajectory_controller()
        if not self._trajectory_controller_client.wait_for_server(timeout_sec=5.0):
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

        send_goal_future = self._trajectory_controller_client.send_goal_async(
            fjt_goal, feedback_callback=feedback_callback
        )
        goal_handle = self._wait_for_future(send_goal_future).result()

        if not goal_handle.accepted:
            self.node.get_logger().error("Trajectory rejected by controller.")
            return False

        self.node.get_logger().debug("Trajectory accepted by controller.")

        # Wait for result while still spinning
        result_future = goal_handle.get_result_async()
        result = self._wait_for_future(result_future).result().result

        if result.error_code == FollowJointTrajectory.Result.SUCCESSFUL:
            self.node.get_logger().debug("Trajectory completed successfully.")
            return True
        else:
            self.node.get_logger().error(
                f"Controller failed with error code {result.error_code}"
            )
            return False

    def _send_joint_velocities(self, joint_velocities: Sequence[float]):
        msg = Float64MultiArray()
        msg.data = list(joint_velocities)
        self._velocity_controller_client.publish(msg)

    def _wait_for_future(self, future):
        while rclpy.ok() and not future.done():
            time.sleep(0.001)
        return future
