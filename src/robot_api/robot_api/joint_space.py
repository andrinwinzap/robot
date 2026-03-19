from typing import List, Sequence

import numpy as np

from robot_api.numeric_kinematics import check_limits


class JointSpace:
    def __init__(self, robot_instance):
        self.robot = robot_instance
        self.max_joint_velocity = 1.0

    def move(self, point: "JointSpace.Point"):
        if not check_limits(point.joint_configuration):
            self.robot.node.get_logger().error(f"Joint positions not within limits")

        path = JointSpace.Path()

        start_point = JointSpace.Point(self.robot._joint_configuration)
        start_point.time_from_start = 0.0
        path.add(start_point)

        dq = np.abs(
            np.array(self.robot._joint_configuration)
            - np.array(point.joint_configuration)
        )
        point.time_from_start = np.max(dq) / self.max_joint_velocity
        path.add(point)

        trajectory = self.robot._generate_trajectory(path)

        return self.robot._send_trajectory(trajectory)
    
    def follow_path(self, path: "JointSpace.Path"):
        
        offset = np.linalg.norm(
            np.array(path.points[0]) - self.robot._joint_configuration
        )

        if offset > 1e-1:
            self.robot.node.get_logger().error("Robot not at start of path")
            return False

        trajectory = self.robot._generate_trajectory(path)

        return self.robot._send_trajectory(trajectory)

    def set_velocities(self, velocities: Sequence[float]):
        self.robot._use_velocity_controller()
        self.robot._send_joint_velocities(velocities)

    def read(self, decimals: int = 5) -> "JointSpace.Point":
        if decimals is None:
            joint_configuration = self.robot._joint_configuration
        else:
            joint_configuration = np.round(self.robot._joint_configuration, decimals)
        return JointSpace.Point(joint_configuration)

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
            return "JointSpace.Point(\n  " + "\n  ".join(joints) + "\n)"

        def __array__(self, dtype=None):
            return np.array(self.joint_configuration, dtype=dtype)

    class Path:
        def __init__(self):
            self.points: List["JointSpace.Point"] = []

        def add(self, point: "JointSpace.Point"):
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
