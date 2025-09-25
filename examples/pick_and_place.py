from robot_api import Robot
import math
import time

########################################
START_X = 0.25
START_Y = 0.5

END_X = 0.2
END_Y = 0.6

PICK_PLACE_HEIGHT = 0.03
TRAVEL_HEIGHT = 0.1

OBJECT_SIZE = 0.02
########################################

robot = Robot()
robot.set_fake_hardware(False)
robot.set_debug_mode(False)

robot.joint_space.speed = 0.1
robot.cartesian_space.speed = 0.03  

robot.tool_changer.attach_tool(robot.tools.gripper)

robot.cartesian_space.move(
    Robot.CartesianSpace.Pose(
        (START_X,START_Y,TRAVEL_HEIGHT), 
        (0,0,0)
        ),
        False
    )

robot.tools.gripper.set_distance(0.05)
time.sleep(1)

robot.cartesian_space.move(
    Robot.CartesianSpace.Pose(
        (START_X,START_Y,PICK_PLACE_HEIGHT), 
        (0,0,0)
        )
    )

robot.tools.gripper.set_distance(OBJECT_SIZE)
time.sleep(1)

robot.cartesian_space.move(
    Robot.CartesianSpace.Pose(
        (START_X,START_Y,TRAVEL_HEIGHT), 
        (0,0,0)
        )
    )

robot.cartesian_space.move(
    Robot.CartesianSpace.Pose(
        (END_X,END_Y,TRAVEL_HEIGHT), 
        (0,0,0)
        )
    )

robot.cartesian_space.move(
    Robot.CartesianSpace.Pose(
        (END_X,END_Y,PICK_PLACE_HEIGHT), 
        (0,0,0)
        )
    )

robot.tools.gripper.set_distance(0.05)
time.sleep(1)

robot.cartesian_space.move(
    Robot.CartesianSpace.Pose(
        (END_X,END_Y,TRAVEL_HEIGHT), 
        (0,0,0)
        )
    )

robot.shutdown()