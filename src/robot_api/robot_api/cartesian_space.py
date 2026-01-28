from typing import List, Sequence

import numpy as np
from scipy.spatial.transform import Rotation as R, Slerp

from robot_api.joint_space import JointSpace
from robot_api.numeric_kinematics import (
    forward_kinematics,
    inverse_kinematics,
    chose_optimal_solution,
    jacobian_dls_pinv,
)


class CartesianSpace:
    def __init__(self, robot_instance):
        self.robot = robot_instance
        self.max_linear_velocity = 0.05
        self.max_angular_velocity = 0.1
        self.max_linear_acceleration = 0.05
        self.interpolation_step_size = 0.01

        self._twist_timer = None
        self._target_twist = np.zeros(6)
        self._twist_active = False

    def move(self, pose: "CartesianSpace.Pose", enforce_linearity: bool = True):
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

        weight_factor = self.max_linear_velocity / self.max_angular_velocity
        effective_distance = max(linear_dist, angular_dist * weight_factor)

        path = CartesianSpace.Path()

        if enforce_linearity:
            s_profile, t_profile = self._trapezoidal_profile(
                effective_distance,
                v_max=self.max_linear_velocity,
                a_max=self.max_linear_acceleration,
                dt=self.interpolation_step_size,
            )

            for s, t in zip(s_profile, t_profile):
                alpha = s / effective_distance if effective_distance > 1e-9 else 1.0
                T = self._interpolate_htm(start, end, alpha)
                pose = CartesianSpace.Pose.from_matrix(T)
                pose.time_from_start = t
                path.add(pose)

        else:
            start_pose = CartesianSpace.Pose.from_matrix(start)
            start_pose.time_from_start = 0.0

            end_pose = pose

            time_linear = linear_dist / self.max_linear_velocity
            time_angular = angular_dist / self.max_angular_velocity

            total_duration = max(time_linear, time_angular)

            end_pose.time_from_start = total_duration

            path.add(start_pose)
            path.add(end_pose)

        return self.follow_path(path)

    def follow_path(self, path: "CartesianSpace.Path"):

        joint_space_path = JointSpace.Path()

        prev_joint_configuration = self.robot._joint_configuration
        for i, pose in enumerate(path):
            T = self._base_to_world() @ pose.as_matrix() @ self._tcp_to_robot()

            ik_solutions = inverse_kinematics(T)
            if not ik_solutions:
                self.robot.node.get_logger().error(f"No IK solution found at pose {i}")
                return False
            prev_joint_configuration = chose_optimal_solution(
                prev_joint_configuration, ik_solutions
            )
            point = JointSpace.Point(prev_joint_configuration)
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

    def twist(
        self, linear_velocity: Sequence[float], angular_velocity: Sequence[float]
    ):
        self._target_twist = np.hstack((linear_velocity, angular_velocity))

        if np.linalg.norm(self._target_twist) < 1e-4:
            self._stop_twist_timer()
            return

        self.robot._use_velocity_controller()
        if not self._twist_active:
            self._twist_active = True
            self._twist_timer = self.robot.node.create_timer(0.01, self._twist_callback)

    def read(self):
        T = (
            self._world_to_base()
            @ forward_kinematics(self.robot._joint_configuration)
            @ self._robot_to_tcp()
        )
        position = T[:3, 3]
        orientation = R.from_matrix(T[:3, :3]).as_euler("xyz")
        pose = CartesianSpace.Pose(position, orientation)
        return pose

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

    def _stop_twist_timer(self):
        if not self._twist_active:
            return

        self._target_twist = np.zeros(6)
        if self._twist_timer:
            self._twist_timer.destroy()
            self._twist_timer = None
        self._twist_active = False
        self.robot._send_joint_velocities(np.zeros(6))

    def _twist_callback(self):
        joint_velocities = (
            jacobian_dls_pinv(self.robot._joint_configuration) @ self._target_twist
        )
        joint_velocities = np.clip(
            joint_velocities,
            -np.array(self.robot.joint_velocity_limits),
            np.array(self.robot.joint_velocity_limits),
        )
        self.robot._send_joint_velocities(joint_velocities)

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
        def from_matrix(cls, T: np.ndarray) -> "CartesianSpace.Pose":
            if T.shape != (4, 4):
                raise ValueError("Input must be a 4x4 homogeneous transform")
            position = T[:3, 3]
            orientation = R.from_matrix(T[:3, :3]).as_euler("xyz")
            return cls(position, orientation)

        def __repr__(self):
            position = np.round(self.position, 5)
            orientation = np.round(self.orientation, 5)
            return (
                f"CartesianSpace.Pose(\n"
                f"  Position:    X={position[0]}, Y={position[1]}, Z={position[2]}\n"
                f"  Orientation: Roll={orientation[0]}, Pitch={orientation[1]}, Yaw={orientation[2]}\n"
                f")"
            )

    class Path:
        def __init__(self):
            self.poses: List["CartesianSpace.Pose"] = []

        def add(self, pose: "CartesianSpace.Pose"):
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
