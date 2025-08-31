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

def chose_optimal_solution(current_joints, ik_solutions):
    current = np.array(current_joints)
    solutions = np.array(ik_solutions)
    diffs = np.linalg.norm(solutions - current, axis=1)
    best_idx = np.argmin(diffs)
    return ik_solutions[best_idx]