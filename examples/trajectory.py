from robot_api import Robot
import math

robot = Robot()
robot.set_fake_hardware(True)

#robot.joint_space.move([-1.60938022, 1.6191366,   0.60254534,  0.0, 0.91991071,  1.53221243])

trajectory = Robot.CartesianSpace.Trajectory()
ori = [0,0,0]

center = [0, 0.1, 0]  # circle center (x, y, z)
radius = 0.02         # circle radius
steps = 50            # number of trajectory points

for i in range(steps):
    theta = 2 * math.pi * i / steps
    pos = [
        center[0] + radius * math.cos(theta),
        center[1] + radius * math.sin(theta),
        center[2]
    ]
    pose = Robot.CartesianSpace.Pose(pos, ori)
    trajectory.add_pose(pose)

robot.cartesian_space.execute_trajectory(trajectory)

robot.shutdown()