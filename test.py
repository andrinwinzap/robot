import time

start = time.time()

from robot_sdk import Robot
import math
from rclpy.logging import LoggingSeverity

print(f"imports took: {time.time()-start}")

start = time.time()

robot = Robot(LoggingSeverity.DEBUG)

print(f"init took: {time.time()-start}")

robot.tool_changer.set_tcp_position([0,0,0.059])
robot.joint_space.set_speed(.2)

robot.cartesian_space.move([-0.25,  0.1,  0.3], [-0.00835864, -0.78326253, -2.34489495])
robot.cartesian_space.move([-0.25,  0.1,  0.1], [-0.00835864, -0.78326253, -2.34489495])
# robot.cartesian_space.move((0.0,0.0,0.1), (0.0, 0.0,0.0))
# robot.cartesian_space.move((0.0,0.0,0.2), (0.0, 0.0,0.0))

robot.shutdown()