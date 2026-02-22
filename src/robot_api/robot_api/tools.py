from std_msgs.msg import Bool, Float32


class Tools:
    def __init__(self, robot_instance):
        self.robot = robot_instance
        self.gripper = self.Gripper(self.robot)

    class Tool:
        def __init__(self, robot_instance):
            self.robot = robot_instance
            self._tcp_position = (0.0, 0.0, 0.0)
            self._tcp_orientation = (1.0, 0.0, 0.0, 0.0)

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


class ToolChanger:
    def __init__(self, robot_instance):
        self.robot = robot_instance
        self.current_tool = None
        self._command_publisher = self.robot.node.create_publisher(
            Bool, "/robot/tool_changer/attach", 10
        )

    def attach_tool(self, tool: Tools.Tool):
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

        self.robot._tcp_position = (0.0, 0.0, 0.0)
        self.robot._tcp_orientation = (1.0, 0.0, 0.0, 0.0)
        self.current_tool = None
