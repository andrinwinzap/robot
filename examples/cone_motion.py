import math
import time
from robot_api import Robot, CartesianSpace

# --- Parameters ---
POSITION = (0.2, 0.5, 0.1)    # Fixed end-effector position (x, y, z) in meters
CONE_ANGLE = math.radians(15)  # Half-angle of the cone
PERIOD = 10.0                   # Time (seconds) to complete one full revolution
ROTATIONS = 2                   # Number of full revolutions

robot = Robot()
robot.set_fake_hardware_mode(False)

# Move directly to the cone starting position (theta=0: roll=0, pitch=CONE_ANGLE).
# Starting here means the twist integration traces a symmetric circle centered
# at (0, 0) in orientation space, giving a fully symmetric cone.
print("Moving to start pose...")
robot.cartesian_space.move(
    CartesianSpace.Pose(POSITION, (0.0, CONE_ANGLE, 0.0)),
    enforce_linearity=False,
)

# Sweep the cone using twist.
# Desired: roll(t) = CONE_ANGLE * sin(omega*t), pitch(t) = CONE_ANGLE * cos(omega*t)
# Starting from (roll=0, pitch=CONE_ANGLE), integrating these velocities gives exactly that.
print("Executing cone motion...")
omega = 2 * math.pi / PERIOD
total_time = PERIOD * ROTATIONS

t_start = time.time()
while True:
    t = time.time() - t_start
    if t >= total_time:
        break
    theta = omega * t

    # Desired Euler angle rates for the circular trajectory
    pitch     =  CONE_ANGLE * math.cos(theta)
    roll_dot  =  CONE_ANGLE * omega * math.cos(theta)
    pitch_dot = -CONE_ANGLE * omega * math.sin(theta)

    # Convert XYZ Euler rates → world-frame angular velocity:
    #   ω_world = [cos(pitch)·roll_dot, pitch_dot, -sin(pitch)·roll_dot]
    # Then rotate to base frame (robot base is 180° around Z, so X/Y flip):
    #   ω_base  = [-cos(pitch)·roll_dot, -pitch_dot, -sin(pitch)·roll_dot]
    cp = math.cos(pitch)
    sp = math.sin(pitch)
    robot.cartesian_space.twist(
        [0.0, 0.0, 0.0],
        [-cp * roll_dot, -pitch_dot, -sp * roll_dot],
    )
    time.sleep(0.01)

robot.cartesian_space.twist([0.0, 0.0, 0.0], [0.0, 0.0, 0.0])

robot.shutdown()
print("Done.")
