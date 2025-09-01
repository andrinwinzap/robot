from robot_api import Robot
import math

robot = Robot()
robot.set_fake_hardware(True)

robot.joint_space.move([-1.60938022, 1.6191366,   0.60254534,  0.0, 0.91991071,  1.53221243])
#robot.joint_space.move([0]*6)
robot.cartesian_space.move((0.0,0.1,0.0), (0, 0.0,0.0))
robot.cartesian_space.move((0.0,0,0), (0, 0.0,0.0))

print(robot.joint_space.get_pose())

robot.shutdown()