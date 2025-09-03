from robot_api import Robot
import math

robot = Robot()
robot.set_fake_hardware(True)

point = Robot.JointSpace.Point([0.0, 0.0,   0.0,  0.0, 0.0,  0.0])

robot.joint_space.move(point)


pose = Robot.CartesianSpace.Pose()

pose.position = (0.0,0.1,0.0)
pose.orientation = (0, 0.0,0.0)

robot.cartesian_space.move(pose, False)

pose.position = (0.0,0,0)
pose.orientation = (0, 0.0,0.0)

robot.cartesian_space.move(pose)

print(robot.joint_space.read())
print(robot.cartesian_space.read())

robot.shutdown()