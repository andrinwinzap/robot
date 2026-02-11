#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import TransformStamped
import tf2_ros
from std_msgs.msg import String
from ament_index_python.packages import get_package_share_directory
from rcl_interfaces.msg import SetParametersResult
from rclpy.qos import QoSProfile, DurabilityPolicy
import os

class ToolManager(Node):
    def __init__(self):
        super().__init__('tool_manager')

        # Declare parameter for selecting active tool
        self.declare_parameter('active_tool', 'gripper')
        self.active_tool = self.get_parameter('active_tool').value

        # Tool frames
        self.tool_frame = 'gripper_slider'   # must match root link in each tool URDF
        self.mount_frame = 'tool_mount'    # frame in robot URDF where tools attach

        # Publisher for RViz
        
        qos = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)

        self.tool_pub = self.create_publisher(String, 'tool_description_topic', qos)

        # TF broadcaster
        self.br = tf2_ros.TransformBroadcaster(self)
        self.timer = self.create_timer(0.05, self.publish_tool_tf)  # 20 Hz

        # Load all tool URDFs at startup
        self.tools = {}
        self.load_all_tools(['gripper', 'welder', 'suction'])  # add all tool names here

        # Publish the initial active tool
        self.publish_active_tool()

        # Parameter callback for switching tools at runtime
        self.add_on_set_parameters_callback(self.on_param_change)

    def load_all_tools(self, tool_names):
        pkg_share = get_package_share_directory('robot_description')
        for tool_name in tool_names:
            urdf_path = os.path.join(pkg_share, 'urdf', f'{tool_name}.urdf')
            try:
                with open(urdf_path, 'r') as f:
                    self.tools[tool_name] = f.read()
                self.get_logger().info(f"Loaded tool URDF: {tool_name}")
            except FileNotFoundError:
                self.get_logger().error(f"Tool URDF not found: {urdf_path}")

    def publish_active_tool(self):
        if self.active_tool in self.tools:
            urdf_string = self.tools[self.active_tool]
            self.tool_pub.publish(String(data=urdf_string))
            self.get_logger().info(f"Published active tool: {self.active_tool}")
        else:
            self.get_logger().error(f"Active tool '{self.active_tool}' not loaded")

    def on_param_change(self, params):
        for p in params:
            if p.name == 'active_tool' and p.type_ == p.Type.STRING:
                if p.value != self.active_tool:
                    self.active_tool = p.value
                    self.publish_active_tool()
        return SetParametersResult(successful=True)


    def publish_tool_tf(self):
        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = self.mount_frame
        t.child_frame_id = self.tool_frame
        t.transform.translation.x = 0.0
        t.transform.translation.y = 0.0
        t.transform.translation.z = 0.0
        t.transform.rotation.x = 0.0
        t.transform.rotation.y = 0.0
        t.transform.rotation.z = 0.0
        t.transform.rotation.w = 1.0
        self.br.sendTransform(t)

def main(args=None):
    rclpy.init(args=args)
    node = ToolManager()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
