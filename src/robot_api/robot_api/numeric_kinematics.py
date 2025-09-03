# kinematics.py

import numpy as np
from .symbolic_kinematics import T_06_func, T_01_func, R_03_func
from .config import EPSILON, JOINT_OFFSETS, LINK_LENGTHS, JOINT_LIMITS

def normalize_angle(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi

def check_limits(joint_angles):
    for angle, (low, high) in zip(joint_angles, JOINT_LIMITS):
        if not (low <= angle <= high):
            return False
    return True

def forward_kinematics(thetas):
    return T_06_func(*thetas)

def inverse_kinematics(T_06):

    R_06 = T_06[:3, :3] # Extract rotation part
    P_06 = T_06[:3, 3] # Extract position part

    P_04 = P_06 - JOINT_OFFSETS["D6"] * R_06[:, 2] # Derive wrist center position

    # Calculate q1
    phi = np.arctan2(JOINT_OFFSETS["D2"], np.sqrt(P_04[0]**2 + P_04[1]**2 - JOINT_OFFSETS["D2"]**2))
    theta_1 = np.arctan2(P_04[1], P_04[0])
    q1 = [theta_1 - phi, theta_1 + (np.pi + phi)]
    
    # Calculate q3
    r = np.sqrt(P_04[0]**2 + P_04[1]**2 - JOINT_OFFSETS["D2"]**2)
    s = P_04[2] - JOINT_OFFSETS["D1"]

    theta_cos = (r**2 + s**2 - LINK_LENGTHS["L2"]**2 - JOINT_OFFSETS["D4"]**2) / (2 * LINK_LENGTHS["L2"] * JOINT_OFFSETS["D4"])

    q3 = [np.arctan2(np.sqrt(1-theta_cos**2), theta_cos), np.arctan2(-np.sqrt(1-theta_cos**2), theta_cos)]

    alpha = np.arctan2(r, s)
    beta = np.arctan2(JOINT_OFFSETS["D4"] * np.sin(q3[0]), LINK_LENGTHS["L2"] + (np.cos(q3[0]) * JOINT_OFFSETS["D4"]))
    q2 = [alpha - beta, alpha + beta]

    T_03_solutions = zip(
    [ q1[0],     q1[0],     q1[1],     q1[1] ],
    [ q2[0],     q2[1],     -q2[0],     -q2[1] ],
    [ q3[0],     q3[1],     -q3[0],     -q3[1] ]
    )
        
    T_06_solutions = []

    for q1, q2, q3 in T_03_solutions:

        # Calculate wrist rotation matrix
        R_03 = R_03_func(q1,q2,q3)
        R_36 = R_03.T @ R_06

        r11, r12, r13 = R_36[0,0], R_36[0,1], R_36[0,2]
        r21, r22, r23 = R_36[1,0], R_36[1,1], R_36[1,2]
        r31, r32, r33 = R_36[2,0], R_36[2,1], R_36[2,2]

        # Extract ZYZ Euler Angles
        q4, q5, q6  = [None, None], [None, None], [None, None]
        q5[0] = np.arccos(r33)

        if abs(np.sin(q5[0])) > EPSILON:
             # nonsingular case
            q4[0] = np.arctan2(r23, r13)
            q6[0] = np.arctan2(r32, -r31)

            q4[1] = (q4[0] + np.pi) % (2*np.pi)
            q5[1] = -q5[0]
            q6[1] = (q6[0] + np.pi) % (2*np.pi)

            T_36_solutions = [
                (q4[0], q5[0], q6[0]),
                (q4[1], q5[1], q6[1])
            ]
        else: 
            # singularity
            q4[0] = 0.0
            if r33 > 0:
                q6[0] = -np.arctan2(r12, r11)
                q5[0] = 0.0
            else:
                q6[0] = np.arctan2(r12, r11)
                q5[0] = np.pi
            T_36_solutions = [(q4[0], q5[0], q6[0])]

        for T_36_solution in T_36_solutions:
            T_06_solution = [normalize_angle(q) for q in [q1, q2, q3, *T_36_solution]]
            if check_limits(T_06_solution):
                T_06_solutions.append(T_06_solution)

    return T_06_solutions

def chose_optimal_solution(current_joints, ik_solutions):
    current = np.array(current_joints)
    solutions = np.array(ik_solutions)
    diffs = np.linalg.norm(solutions - current, axis=1)
    best_idx = np.argmin(diffs)
    return ik_solutions[best_idx]