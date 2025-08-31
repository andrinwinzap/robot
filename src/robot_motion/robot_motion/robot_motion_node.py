import numpy as np

import rclpy
from rclpy.action import ActionClient, ActionServer
from rclpy.node import Node

from control_msgs.action import FollowJointTrajectory
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

from scipy.spatial.transform import Rotation as R, Slerp
from scipy.interpolate import CubicSpline

from robot_motion_interfaces.action import CartesianSpaceMotion, JointSpaceMotion
from robot_motion_interfaces.srv import GetCartesianSpacePose, GetJointSpacePose

from robot_motion.kinematics import forward_kinematics, inverse_kinematics
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

        self.declare_parameter("joint_space_speed", 1.0)       # rad/s
        self.declare_parameter("cartesian_space_speed", 0.05)   # m/s
        self.declare_parameter("num_waypoints", 50)
        self.declare_parameter("tcp_position",    [0.0, 0.0, 0.0])
        self.declare_parameter("tcp_orientation", [1.0, 0.0, 0.0, 0.0])
        self.declare_parameter("robot_position",    [0.0, 0.35, 0.0])
        self.declare_parameter("robot_orientation", [0.0, 0.0, 0.0, 1.0])
        self.declare_parameter("joint_velocity_limits", [5.0, 5.0, 5.0, 5.0, 5.0, 5.0])       # rad/s
        self.declare_parameter("joint_acceleration_limits", [3.14, 3.14, 3.14, 3.14, 3.14, 3.14])   # rad/s^2

        
        self.trajectory_client.wait_for_server()
        self.get_logger().info("Robot kinematics node ready.")

    def robot_to_tcp(self):
        pos_param = list(self.get_parameter("tcp_position").value)
        quat_param = list(self.get_parameter("tcp_orientation").value)

        if not (isinstance(pos_param, list) and len(pos_param) == 3):
            self.get_logger().warn("tcp_position must be a list of 3 floats [x, y, z]")
            return np.eye(4)

        if not (isinstance(quat_param, list) and len(quat_param) == 4):
            self.get_logger().warn("tcp_orientation must be a list of 4 floats [x, y, z, w]")
            return np.eye(4)

        pos = np.array(pos_param, float)
        quat = np.array(quat_param, float)

        if np.linalg.norm(quat) == 0:
            self.get_logger().warn("tcp_orientation quaternion must not be zero.")
            return np.eye(4)

        quat /= np.linalg.norm(quat)

        T = np.eye(4)
        T[:3, :3] = R.from_quat(quat).as_matrix()
        T[:3, 3] = pos
        return T
    
    def tcp_to_robot(self):
        return np.linalg.inv(self.robot_to_tcp())

    def world_to_base(self):
        pos = np.array(self.get_parameter("robot_position").value, float)
        quat = np.array(self.get_parameter("robot_orientation").value, float)
        quat /= np.linalg.norm(quat)
        T = np.eye(4)
        T[:3,:3] = R.from_quat(quat).as_matrix()
        T[:3, 3] = pos
        return T
    
    def base_to_world(self):
        return np.linalg.inv(self.world_to_base())
    
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

    def interpolate_pose(self, start_T, end_T, t):
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

    def generate_cubic_spline(self, points, proposed_time):
        points = np.array(points)
        if points.shape[1] != 6:
            raise ValueError("Points must have 6 dimensions (joints).")
        
        max_vel = np.array(self.get_parameter("joint_velocity_limits").value)
        max_acc = np.array(self.get_parameter("joint_acceleration_limits").value)
        
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
            
            vel_ratio = max_joint_vel / max(max_vel)
            acc_ratio = max_joint_acc / max(max_acc)
            scale_factor = max(vel_ratio, np.sqrt(acc_ratio), 1.0)  # at least 1
            
            if scale_factor <= 1.0:
                break
            actual_time *= scale_factor
        
        return splines, actual_time

    def generate_trajectory(self, points, proposed_time):
        trajectory = JointTrajectory()
        trajectory.header.stamp = self.get_clock().now().to_msg()
        trajectory.joint_names = self.joint_names

        splines, actual_time = self.generate_cubic_spline(points, proposed_time)

        num_points = self.get_parameter("num_waypoints").value
        times = np.linspace(0, actual_time, num_points)

        for t in times:
            point = JointTrajectoryPoint()
            point.positions = [s(t, 0) for s in splines]
            point.velocities = [s(t, 1) for s in splines]
            point.accelerations = [s(t, 2) for s in splines]
            point.time_from_start.sec = int(t)
            point.time_from_start.nanosec = int((t % 1) * 1e9)
            trajectory.points.append(point)

        trajectory.points[-1].velocities = [0.0] * len(self.joint_names)
        trajectory.points[-1].accelerations = [0.0] * len(self.joint_names)

        return trajectory
    
    def cartesian_space_pose_getter_callback(self, request, response):
        if self.current_joint_positions is None:
            self.get_logger().warn("No joint states available to compute pose.")
            empty_pose = PoseStamped()
            empty_pose.header.stamp = self.get_clock().now().to_msg()
            empty_pose.header.frame_id = "world"
            response.pose = empty_pose
            return response

        T = self.world_to_base() @ forward_kinematics(self.current_joint_positions) @ self.robot_to_tcp()

        pose_stamped = PoseStamped()
        pose_stamped.header.stamp = self.get_clock().now().to_msg()
        pose_stamped.header.frame_id = "world"
        pose_stamped.pose.position.x = T[0, 3]
        pose_stamped.pose.position.y = T[1, 3]
        pose_stamped.pose.position.z = T[2, 3]
        quat = R.from_matrix(T[:3, :3]).as_quat()
        pose_stamped.pose.orientation.x = quat[0]
        pose_stamped.pose.orientation.y = quat[1]
        pose_stamped.pose.orientation.z = quat[2]
        pose_stamped.pose.orientation.w = quat[3]

        response.pose = pose_stamped
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

        tcp_world_T = self.pose_to_transform(goal.pose.pose)
        if tcp_world_T is None:
            goal_handle.abort()
            return CartesianSpaceMotion.Result(success=False, message="Invalid target pose.")

        end_T = self.base_to_world() @ tcp_world_T @ self.tcp_to_robot()

        start_T = forward_kinematics(self.current_joint_positions)
        
        num_points = self.get_parameter("num_waypoints").value

        points=[]

        prev_joints = self.current_joint_positions
        for i in range(num_points):
            alpha = i/(num_points - 1)
            T = self.interpolate_pose(start_T, end_T, alpha)
            ik_solutions = [sol for sol in inverse_kinematics(T) if check_limits(sol)]
            if not ik_solutions:
                goal_handle.abort()
                return CartesianSpaceMotion.Result(success=False, message=f"No IK solution found at alpha={alpha}")
            prev_joints = chose_optimal_solution(prev_joints, ik_solutions)
            points.append(prev_joints)

        start_pos = np.array(start_T[:3, 3])
        end_pos = np.array(end_T[:3, 3])
        dist = np.linalg.norm(end_pos - start_pos)

        cartesian_space_speed = self.get_parameter("cartesian_space_speed").value
        time_cartesian_space = dist / cartesian_space_speed if cartesian_space_speed > 0 else 5.0

        trajectory = self.generate_trajectory(points, time_cartesian_space)

        return await self.send_trajectory(
            trajectory=trajectory,
            goal_handle=goal_handle,
            result_type=CartesianSpaceMotion.Result,
            feedback_type=CartesianSpaceMotion.Feedback
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

        points = np.array([np.array(self.current_joint_positions), np.array(end_joints)])

        joint_space_speed = self.get_parameter("joint_space_speed").value
        dq = np.abs(np.array(self.current_joint_positions) - np.array(end_joints))
        proposed_time = np.max(dq) / joint_space_speed

        trajectory = self.generate_trajectory(points, proposed_time)

        return await self.send_trajectory(
            trajectory=trajectory,
            goal_handle=goal_handle,
            result_type=JointSpaceMotion.Result,
            feedback_type=JointSpaceMotion.Feedback
        )

    async def send_trajectory(self, trajectory, goal_handle, result_type, feedback_type):
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
