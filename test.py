from robot_sdk import Robot
import math

robot = Robot()

robot.tool_changer.set_tcp_position([0,0,0.059])
robot.cartesian_space.set_speed(0.2)
robot.joint_space.set_speed(2)

robot.joint_space.move((math.pi/2,0,0,0,0,0,0))
robot.joint_space.move((-math.pi/2,0,0,0,0,0,0))
# robot.cartesian_space.move((0.0,0.0,0.1), (0.0, 0.0,0.0))
# robot.cartesian_space.move((0.0,0.0,0.2), (0.0, 0.0,0.0))

robot.shutdown()