import numpy as np

import rclpy
from rclpy.action import ActionClient, ActionServer
from rclpy.node import Node

from control_msgs.action import FollowJointTrajectory
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

from scipy.spatial.transform import Rotation as R

from robot_motion_interfaces.action import CartesianSpaceMotion, JointSpaceMotion
from robot_motion_interfaces.srv import GetCartesianSpacePose, GetJointSpacePose

from robot_motion.robot_motion import forward_kinematics, inverse_kinematics
from robot_motion.utills import check_limits, chose_optimal_solution

class KinematicsNode(Node):
    def __init__(self):
        super().__init__('robot_motion_node')

        self.joint_names = [f"joint_{i+1}" for i in range(6)]
        self.current_joint_positions = None

        self.trajectory_client = ActionClient(self, FollowJointTrajectory, '/joint_trajectory_controller/follow_joint_trajectory')
        self.create_subscription(JointState, '/joint_states', self.joint_states_callback, 10)
        
        self.create_service(GetCartesianSpacePose, '/robot_motion/cartesian_space/get_pose', self.cartesian_space_pose_getter_callback)
        self.create_service(GetJointSpacePose, '/robot_motion/joint_space/get_pose', self.joint_space_pose_getter_callback)
        
        self.cartesian_space_motion_server = ActionServer(
            self,
            CartesianSpaceMotion,
            '/robot_motion/cartesian_space/motion',
            self.cartesian_space_motion_callback
        )

        self.joint_space_motion_server = ActionServer(
            self,
            JointSpaceMotion,
            '/robot_motion/joint_space/motion',
            self.joint_space_motion_callback
        )

        self.declare_parameter("interpolation_type", "cubic")
        self.declare_parameter("joint_space_speed", 1.0)       # rad/s
        self.declare_parameter("cartesian_space_speed", 0.05)   # m/s
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

    async def cartesian_space_motion_callback(self, goal_handle):
        self.get_logger().info("Received Cartesian goal request.")
        goal = goal_handle.request

        if self.current_joint_positions is None:
            goal_handle.abort()
            return CartesianSpaceMotion.Result(success=False, message="No joint state received yet.")

        end_T = self.pose_to_transform(goal.pose.pose)
        if end_T is None:
            goal_handle.abort()
            return CartesianSpaceMotion.Result(success=False, message="Invalid target pose.")

        ik_solutions = inverse_kinematics(end_T)
        if not ik_solutions:
            goal_handle.abort()
            return CartesianSpaceMotion.Result(success=False, message="No IK solution found.")

        end_joints = chose_optimal_solution(self.current_joint_positions, ik_solutions)

        start_T = forward_kinematics(self.current_joint_positions)
        start_pos = np.array(start_T[:3, 3])
        end_pos = np.array(end_T[:3, 3])
        dist = np.linalg.norm(end_pos - start_pos)

        cartesian_space_speed = self.get_parameter("cartesian_space_speed").value
        total_time = dist / cartesian_space_speed if cartesian_space_speed > 0 else 5.0

        return await self.send_trajectory(
            start=self.current_joint_positions,
            end=end_joints,
            goal_handle=goal_handle,
            result_type=CartesianSpaceMotion.Result,
            feedback_type=CartesianSpaceMotion.Feedback,
            total_time=total_time
        )

    async def joint_space_motion_callback(self, goal_handle):
        self.get_logger().info("Received Joint goal request.")
        goal = goal_handle.request

        if self.current_joint_positions is None:
            goal_handle.abort()
            return JointSpaceMotion.Result(success=False, message="No joint state received yet.")

        joint_map = dict(zip(goal.joint_state.name, goal.joint_state.position))
        try:
            end_joints = [joint_map[name] for name in self.joint_names]
        except KeyError as e:
            goal_handle.abort()
            return JointSpaceMotion.Result(success=False, message=f"Missing joint: {e}")

        if not check_limits(end_joints):
            goal_handle.abort()
            return JointSpaceMotion.Result(success=False, message="Joint limits exceeded.")

        joint_space_speed = self.get_parameter("joint_space_speed").value
        dq = np.abs(np.array(end_joints) - np.array(self.current_joint_positions))
        max_dq = np.max(dq)
        total_time = max_dq / joint_space_speed if joint_space_speed > 0 else 5.0

        return await self.send_trajectory(
            start=self.current_joint_positions,
            end=end_joints,
            goal_handle=goal_handle,
            result_type=JointSpaceMotion.Result,
            feedback_type=JointSpaceMotion.Feedback,
            total_time=total_time
        )

    def create_trajectory(self, start, end, total_time, num_points, interpolation):
        trajectory = JointTrajectory()
        trajectory.header.stamp = self.get_clock().now().to_msg()
        trajectory.joint_names = self.joint_names

        for i in range(num_points):
            t_norm = i / (num_points - 1)
            pos, vel, acc = self.interpolate_joint_trajectory(
                np.array(start),
                np.array(end),
                t_norm,
                total_time,
                interpolation
            )
            point = JointTrajectoryPoint()
            point.positions = pos.tolist()
            point.velocities = vel.tolist()
            point.accelerations = acc.tolist()
            point.time_from_start.sec = int(total_time * t_norm)
            point.time_from_start.nanosec = int((total_time * t_norm % 1) * 1e9)
            trajectory.points.append(point)

        # Ensure last point has zero velocity and acceleration
        trajectory.points[-1].velocities = [0.0] * len(self.joint_names)
        trajectory.points[-1].accelerations = [0.0] * len(self.joint_names)
        return trajectory

    async def send_trajectory(self, start, end, goal_handle, result_type, feedback_type, total_time):
        num_points = self.get_parameter("num_waypoints").value
        interpolation = self.get_parameter("interpolation_type").value

        trajectory = self.create_trajectory(start, end, total_time, num_points, interpolation)

        fjt_client = ActionClient(self, FollowJointTrajectory, '/joint_trajectory_controller/follow_joint_trajectory')
        if not fjt_client.wait_for_server(timeout_sec=5.0):
            goal_handle.abort()
            return result_type(success=False, message="FollowJointTrajectory server not available.")

        fjt_goal = FollowJointTrajectory.Goal()
        fjt_goal.trajectory = trajectory

        def feedback_callback(feedback_msg):
            traj_feedback = feedback_msg.feedback
            feedback = feedback_type()
            feedback.desired_positions = list(traj_feedback.desired.positions or [])
            feedback.desired_velocities = list(traj_feedback.desired.velocities or [])
            feedback.actual_positions = list(traj_feedback.actual.positions or [])
            feedback.actual_velocities = list(traj_feedback.actual.velocities or [])
            goal_handle.publish_feedback(feedback)

        send_goal_future = fjt_client.send_goal_async(fjt_goal, feedback_callback=feedback_callback)
        await send_goal_future
        fjt_goal_handle = send_goal_future.result()

        if not fjt_goal_handle.accepted:
            goal_handle.abort()
            return result_type(success=False, message="Trajectory rejected by controller.")

        self.get_logger().info("Trajectory accepted by controller.")
        result_future = fjt_goal_handle.get_result_async()
        await result_future
        result = result_future.result().result

        if result.error_code == FollowJointTrajectory.Result.SUCCESSFUL:
            goal_handle.succeed()
            return result_type(success=True, message="Motion completed successfully.")
        else:
            goal_handle.abort()
            return result_type(success=False, message=f"Controller failed with error code {result.error_code}")

def main(args=None):
    rclpy.init(args=args)
    node = KinematicsNode()
    rclpy.spin(node)
    rclpy.shutdown()
