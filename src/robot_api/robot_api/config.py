import numpy as np

EPSILON = 1e-6

LINK_LENGTHS = {"L2": 0.2}

JOINT_OFFSETS = {"D1": 0.182, "D2": 0.0135, "D4": 0.1885, "D6": 0.05813392}

# TODO: Get limits from urdf
JOINT_POSITION_LIMITS = [
    (-np.pi, np.pi),
    (-np.pi, 0),  # Prevent shoulder from flipping over
    (-np.pi, np.pi),
    (-np.pi, np.pi),
    (-np.pi / 2, np.pi / 2),
    (-np.pi, np.pi),
]

JOINT_VELOCITY_LIMITS = [
    np.pi * 5,
    np.pi * 5,
    np.pi * 5,
    np.pi * 5,
    np.pi * 5,
    np.pi * 5,
]

JOINT_ACCELERATION_LIMITS = [
    np.pi * 5,
    np.pi * 5,
    np.pi * 5,
    np.pi * 5,
    np.pi * 5,
    np.pi * 5,
]

ROBOT_POSITION = (0.0, 0.35, 0.0)
ROBOT_ORIENTATION = (0.0, 0.0, 1.0, 0.0)

JOINT_TRAJECTORY_RESOLUTION = 50
