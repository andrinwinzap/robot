import numpy as np
from robot_api import Robot

robot = Robot()
robot.set_fake_hardware(False)

robot.tool_changer.attach_tool(robot.tools.gripper)

x0 = 0.175
y0 = 0.2
z0 = 0.05

wave_length = 0.3
num_periods = 2
wave_amplitude = 0.03
num_points = 50
total_time = 10

robot.cartesian_space.move(
Robot.CartesianSpace.Pose(
    (x0,y0,z0), 
    (0,0,0)
    )
)

path = Robot.CartesianSpace.Path()
for i in range(num_points):
    alpha = i / (num_points - 1)
    y = y0 + alpha * wave_length
    x = x0 + wave_amplitude * np.sin(2 * np.pi * alpha * num_periods)
    z = z0

    pose = Robot.CartesianSpace.Pose(
        position=(x, y, z),
        orientation=(0,0,0),
        time_from_start=alpha*total_time
    )
    path.add(pose)

robot.cartesian_space.follow_path(path)

print(robot.cartesian_space.read())

robot.shutdown()
