from robot_api import Robot


# Initialize the Robot API
robot = Robot()

# Set API Options
robot.set_fake_hardware(False)
robot.set_debug_mode(False)

# Set motion speeds
robot.joint_space.speed = 0.1
robot.cartesian_space.speed = 0.05

# Attach the gripper tool
robot.tool_changer.attach_tool(robot.tools.gripper)

# Control the gripper
robot.tools.gripper.set_distance(0.05)

# Move the robot in Cartesian space
robot.cartesian_space.move(
    Robot.CartesianSpace.Pose((100, 100, 100), (0, 0, 0)), enforce_linearity=False
)

# Move the robot in joint space
robot.joint_space.move(Robot.JointSpace.Point([0, 0, 0, 0, 0, 0]))

# Shut down the robot
robot.shutdown()
