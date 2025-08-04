import rclpy
from rclpy.node import Node
from rclpy.time import Duration
from rclpy.action import ActionClient
from control_msgs.action import FollowJointTrajectory
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from builtin_interfaces.msg import Duration as BuiltinDuration
from rclpy.task import Future

from sensor_msgs.msg import JointState
from geometry_msgs.msg import PoseStamped
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

from robot_motion_interfaces.srv import GetCartesianSpacePose, GetJointSpacePose

from robot_motion.robot_motion import forward_kinematics, inverse_kinematics

from robot_motion.utills import check_limits

import numpy as np
from scipy.spatial.transform import Rotation as R, Slerp

def choose_min_movement_solution(current_joints, ik_solutions):
    current = np.array(current_joints)
    solutions = np.array(ik_solutions)
    diffs = np.linalg.norm(solutions - current, axis=1)  # Euclidean distance in joint space
    best_idx = np.argmin(diffs)
    return ik_solutions[best_idx]

class KinematicsNode(Node):
    def __init__(self):
        super().__init__('robot_motion_node')

        self.joint_names = [f"joint_{i+1}" for i in range(6)]
        self.current_joint_positions = None

        self.trajectory_client = ActionClient(self, FollowJointTrajectory, '/joint_trajectory_controller/follow_joint_trajectory')
        self.create_subscription(JointState, '/joint_states', self.joint_states_callback, 10)
        
        self.create_service(GetCartesianSpacePose, '/robot_motion/cartesian_space/get_pose', self.cartesian_space_pose_getter_callback)
        self.create_service(GetJointSpacePose, '/robot_motion/joint_space/get_pose', self.joint_space_pose_getter_callback)
        
        self.create_subscription(PoseStamped, '/robot_motion/cartesian_space/set_goal_pose', self.cartesian_space_goal_pose_setter_callback, 10)
        self.create_subscription(JointState, '/robot_motion/joint_space/set_goal_pose', self.joint_space_goal_pose_setter_callback, 10)

        self.declare_parameter("interpolation_type", "cubic")
        self.declare_parameter("total_time", 5.0)
        self.declare_parameter("num_waypoints", 50)

        self.trajectory_client.wait_for_server()
        self.get_logger().info("Robot kinematics node ready.")

    def joint_states_callback(self, msg: JointState):
        joint_map = dict(zip(msg.name, msg.position))
        try:
            self.current_joint_positions = [joint_map[name] for name in self.joint_names]
        except KeyError as e:
            self.get_logger().warn(f"Missing joint in /joint_states input: {e}")
            return

    def cartesian_space_goal_pose_setter_callback(self, msg: PoseStamped):
        if self.current_joint_positions is None:
            self.get_logger().warn("No joint state received yet.")
            return

        start_T = forward_kinematics(self.current_joint_positions)
        start_pos = start_T[:3, 3]
        start_ori = R.from_matrix(start_T[:3, :3])

        end_T = self.pose_to_transform(msg.pose)
        if end_T is None:
            self.get_logger().warn("Invalid target pose received. Skipping trajectory generation.")
            return
        end_pos = end_T[:3, 3]
        end_ori = R.from_matrix(end_T[:3, :3])

        ik_solutions = inverse_kinematics(end_T)
        if not ik_solutions:
            self.get_logger().warn("No IK solution for target pose.")
            return

        end_joints = choose_min_movement_solution(self.current_joint_positions, ik_solutions)


        total_time = self.get_parameter("total_time").value
        num_points = self.get_parameter("num_waypoints").value
        interpolation = self.get_parameter("interpolation_type").value

        trajectory = JointTrajectory()
        trajectory.header.stamp = self.get_clock().now().to_msg()
        trajectory.joint_names = self.joint_names

        # Setup slerp for orientation interpolation
        key_times = [0, 1]
        key_rots = R.from_quat([start_ori.as_quat(), end_ori.as_quat()])
        slerp = Slerp(key_times, key_rots)

        for i in range(num_points):
            t_norm = i / (num_points - 1)

            # Interpolate joint positions with the existing interpolation method
            pos, vel, acc = self.interpolate_joint_trajectory(
                np.array(self.current_joint_positions),
                np.array(end_joints),
                t_norm,
                total_time,
                interpolation
            )

            point = JointTrajectoryPoint()
            point.positions = pos.tolist()
            point.velocities = vel.tolist()
            point.accelerations = acc.tolist()
            point.time_from_start = Duration(seconds=(total_time * t_norm)).to_msg()

            trajectory.points.append(point)

        # zero final velocity & acceleration
        trajectory.points[-1].velocities = [0.0] * 6
        trajectory.points[-1].accelerations = [0.0] * 6

        self.send_trajectory_goal(trajectory)
        self.get_logger().info(f"Published trajectory with {num_points} points.")

    def pose_to_transform(self, pose):
        quat = [pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w]
        if np.isclose(np.linalg.norm(quat), 0.0):
            self.get_logger().warn("Received pose with zero-norm quaternion. Ignoring pose.")
            return None

        r = R.from_quat(quat)
        T = np.eye(4)
        T[:3, :3] = r.as_matrix()
        T[:3, 3] = [pose.position.x, pose.position.y, pose.position.z]
        return T

    def transform_to_pose(self, T):
        pose = PoseStamped()
        pose.header.stamp = self.get_clock().now().to_msg()
        pose.header.frame_id = "base_link"
        pose.pose.position.x = T[0, 3]
        pose.pose.position.y = T[1, 3]
        pose.pose.position.z = T[2, 3]
        quat = R.from_matrix(T[:3, :3]).as_quat()
        pose.pose.orientation.x = quat[0]
        pose.pose.orientation.y = quat[1]
        pose.pose.orientation.z = quat[2]
        pose.pose.orientation.w = quat[3]
        return pose

    def interpolate_joint_trajectory(self, q0, qf, t, T, mode):
        dq = qf - q0
        pos = np.zeros_like(q0)
        vel = np.zeros_like(q0)
        acc = np.zeros_like(q0)
        t_scaled = t * T

        if mode == "linear":
            pos = q0 + dq * t
            vel = dq / T
            acc = np.zeros_like(q0)
        elif mode == "cubic":
            a0 = q0
            a1 = 0
            a2 = 3 * dq / T**2
            a3 = -2 * dq / T**3
            pos = a0 + a2 * t_scaled**2 + a3 * t_scaled**3
            vel = 2 * a2 * t_scaled + 3 * a3 * t_scaled**2
            acc = 2 * a2 + 6 * a3 * t_scaled
        elif mode == "quintic":
            a0 = q0
            a1 = 0
            a2 = 0
            a3 = 10 * dq / T**3
            a4 = -15 * dq / T**4
            a5 = 6 * dq / T**5
            t2 = t_scaled**2
            t3 = t2 * t_scaled
            t4 = t3 * t_scaled
            t5 = t4 * t_scaled
            pos = a0 + a3 * t3 + a4 * t4 + a5 * t5
            vel = 3 * a3 * t2 + 4 * a4 * t3 + 5 * a5 * t4
            acc = 6 * a3 * t_scaled + 12 * a4 * t2 + 20 * a5 * t3
        else:
            pos = q0 + dq * t
            vel = dq / T
            acc = np.zeros_like(q0)

        return pos, vel, acc
    
    def cartesian_space_pose_getter_callback(self, request, response):
        if self.current_joint_positions is None:
            self.get_logger().warn("No joint states available to compute pose.")
            empty_pose = PoseStamped()
            empty_pose.header.stamp = self.get_clock().now().to_msg()
            empty_pose.header.frame_id = "base_link"
            response.pose = empty_pose
        else:
            T = forward_kinematics(self.current_joint_positions)
            pose_stamped = self.transform_to_pose(T)  # this returns a PoseStamped
            response.pose = pose_stamped  # Assign full PoseStamped, not just Pose
        return response
    
    def joint_space_pose_getter_callback(self, request, response):
        if self.current_joint_positions is None:
            self.get_logger().warn("No joint state available to respond.")
            response.joint_names = []
            response.joint_positions = []
        else:
            response.joint_names = self.joint_names
            response.joint_positions = self.current_joint_positions
        return response

    def joint_space_goal_pose_setter_callback(self, msg: JointState):
        if self.current_joint_positions is None:
            self.get_logger().warn("No joint state received yet to plan from.")
            return

        joint_map = dict(zip(msg.name, msg.position))
        try:
            target_positions = [joint_map[name] for name in self.joint_names]
        except KeyError as e:
            self.get_logger().warn(f"Missing joint in joint_goal input: {e}")
            return

        # Check joint limits using your utility function
        if not check_limits(target_positions):
            self.get_logger().warn("Requested joint positions exceed joint limits. Ignoring command.")
            return

        # Proceed with trajectory generation as before
        total_time = self.get_parameter("total_time").value
        num_points = self.get_parameter("num_waypoints").value
        interpolation = self.get_parameter("interpolation_type").value

        trajectory = JointTrajectory()
        trajectory.header.stamp = self.get_clock().now().to_msg()
        trajectory.joint_names = self.joint_names

        for i in range(num_points):
            t_norm = i / (num_points - 1)
            pos, vel, acc = self.interpolate_joint_trajectory(
                np.array(self.current_joint_positions),
                np.array(target_positions),
                t_norm,
                total_time,
                interpolation
            )
            point = JointTrajectoryPoint()
            point.positions = pos.tolist()
            point.velocities = vel.tolist()
            point.accelerations = acc.tolist()
            point.time_from_start = Duration(seconds=(total_time * t_norm)).to_msg()
            trajectory.points.append(point)

        trajectory.points[-1].velocities = [0.0] * len(self.joint_names)
        trajectory.points[-1].accelerations = [0.0] * len(self.joint_names)

        self.send_trajectory_goal(trajectory)
        self.get_logger().info(f"Published trajectory to joint goal with {num_points} points.")

    def send_trajectory_goal(self, trajectory: JointTrajectory):
        if not self.trajectory_client.wait_for_server(timeout_sec=2.0):
            self.get_logger().error("FollowJointTrajectory action server not available.")
            return

        goal_msg = FollowJointTrajectory.Goal()
        goal_msg.trajectory = trajectory
        goal_msg.goal_time_tolerance = BuiltinDuration(sec=1)  # Optional

        self.get_logger().info("Sending trajectory goal to FollowJointTrajectory action server...")

        send_goal_future = self.trajectory_client.send_goal_async(goal_msg)

        def goal_response_callback(future: Future):
            goal_handle = future.result()
            if not goal_handle.accepted:
                self.get_logger().warn("Trajectory goal was rejected.")
                return

            self.get_logger().info("Trajectory goal accepted.")
            get_result_future = goal_handle.get_result_async()

            def result_callback(future: Future):
                result = future.result().result
                self.get_logger().info(f"Trajectory execution finished with status: {result.error_code}")
            get_result_future.add_done_callback(result_callback)

        send_goal_future.add_done_callback(goal_response_callback)

def main(args=None):
    rclpy.init(args=args)
    node = KinematicsNode()
    rclpy.spin(node)
    rclpy.shutdown()
