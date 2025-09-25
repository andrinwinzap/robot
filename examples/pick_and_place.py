from robot_api import Robot
import math
import time

robot = Robot()
robot.set_fake_hardware(False)
robot.set_debug_mode(False)

robot.joint_space.speed = 0.1
robot.cartesian_space.speed = 0.03  

robot.tool_changer.attach_tool(robot.tools.gripper)

robot.cartesian_space.move(
    Robot.CartesianSpace.Pose(
        (0.2,0.5,0.1), 
        (0,0,0)
        ),
        False
    )

robot.tools.gripper.set_distance(0.05)
time.sleep(1)

robot.cartesian_space.move(
    Robot.CartesianSpace.Pose(
        (0.2,0.5,0.03), 
        (0,0,0)
        )
    )

robot.tools.gripper.set_distance(0.00)
time.sleep(1)

robot.cartesian_space.move(
    Robot.CartesianSpace.Pose(
        (0.2,0.5,0.1), 
        (0,0,0)
        )
    )

robot.cartesian_space.move(
    Robot.CartesianSpace.Pose(
        (0.2,0.6,0.1), 
        (0,0,0)
        )
    )

robot.cartesian_space.move(
    Robot.CartesianSpace.Pose(
        (0.2,0.6,0.03), 
        (0,0,0)
        )
    )

robot.tools.gripper.set_distance(0.05)
time.sleep(1)

robot.cartesian_space.move(
    Robot.CartesianSpace.Pose(
        (0.2,0.6,0.1), 
        (0,0,0)
        )
    )

robot.shutdown()