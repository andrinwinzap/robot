from robot_api import Robot
import math

robot = Robot()
robot.set_fake_hardware(True)

point = Robot.JointSpace.Point([-1.60938022, 1.6191366,   0.60254534,  0.0, 0.91991071,  1.53221243])

robot.joint_space.move(point)


pose = Robot.CartesianSpace.Pose()

pose.position = (0.0,0.1,0.0)
pose.orientation = (0, 0.0,0.0)

robot.cartesian_space.move(pose, False)

pose.position = (0.0,0,0)
pose.orientation = (0, 0.0,0.0)

robot.cartesian_space.move(pose, False)

print(robot.joint_space.read())
print(robot.cartesian_space.read())

robot.shutdown()