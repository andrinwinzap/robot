# utils.py

import numpy as np
from .config import JOINT_LIMITS

def check_limits(joint_angles):
    for angle, (low, high) in zip(joint_angles, JOINT_LIMITS):
        if not (low <= angle <= high):
            return False
    return True

def normalize_angle(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi