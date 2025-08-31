from robot_sdk import Robot
import math
from rclpy.logging import LoggingSeverity

robot = Robot(LoggingSeverity.DEBUG)
robot.set_simulation_mode(True)
robot.tool_changer.set_tcp_position([0,0,0])
robot.joint_space.set_speed(.2)

robot.cartesian_space.move((0.0,0.1,0.0), (0.0, 0.0,0.0))
robot.cartesian_space.move((0.0,0,0), (0.0, 0.0,0.0))

print(robot.cartesian_space.get_pose())

robot.shutdown()