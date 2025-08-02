# kinematics_node.py
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Header

from robot_kinematics import forward_kinematics, inverse_kinematics
import numpy as np

class KinematicsNode(Node):
    def __init__(self):
        super().__init__('robot_kinematics_node')

        self.fk_pub = self.create_publisher(PoseStamped, 'fk_out', 10)
        self.ik_pub = self.create_publisher(JointState, 'ik_out', 10)

        self.create_subscription(JointState, 'fk_in', self.fk_callback, 10)
        self.create_subscription(PoseStamped, 'ik_in', self.ik_callback, 10)

        self.get_logger().info("Robot kinematics node ready.")

    def fk_callback(self, msg: JointState):
        if len(msg.position) < 6:
            self.get_logger().warn("FK input must have 6 joint values.")
            return

        T_06 = forward_kinematics(msg.position)

        pose = PoseStamped()
        pose.header = Header()
        pose.header.stamp = self.get_clock().now().to_msg()
        pose.header.frame_id = "base_link"

        pose.pose.position.x = T_06[0, 3]
        pose.pose.position.y = T_06[1, 3]
        pose.pose.position.z = T_06[2, 3]

        # Convert rotation matrix to quaternion
        from scipy.spatial.transform import Rotation as R
        quat = R.from_matrix(T_06[:3, :3]).as_quat()
        pose.pose.orientation.x = quat[0]
        pose.pose.orientation.y = quat[1]
        pose.pose.orientation.z = quat[2]
        pose.pose.orientation.w = quat[3]

        self.fk_pub.publish(pose)

    def ik_callback(self, msg: PoseStamped):
        from scipy.spatial.transform import Rotation as R

        pos = msg.pose.position
        ori = msg.pose.orientation
        r = R.from_quat([ori.x, ori.y, ori.z, ori.w])
        T_06 = np.eye(4)
        T_06[:3, :3] = r.as_matrix()
        T_06[:3, 3] = [pos.x, pos.y, pos.z]

        solutions = inverse_kinematics(T_06)
        if not solutions:
            self.get_logger().warn("No IK solutions found.")
            return

        for sol in solutions:
            js = JointState()
            js.header = Header()
            js.header.stamp = self.get_clock().now().to_msg()
            js.name = [f"joint_{i+1}" for i in range(6)]
            js.position = sol
            self.ik_pub.publish(js)

def main(args=None):
    rclpy.init(args=args)
    node = KinematicsNode()
    rclpy.spin(node)
    rclpy.shutdown()
