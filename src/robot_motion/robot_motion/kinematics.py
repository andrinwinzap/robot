# robot_motion.py

import numpy as np
from .symbolic_kinematics import T_06_func, T_01_func, R_03_func
from .utills import check_limits, normalize_angle
from .config import EPSILON, JOINT_OFFSETS, LINK_LENGTHS

def forward_kinematics(thetas):
    return T_06_func(*thetas)

def inverse_kinematics(T_06):
    R_06 = T_06[:3, :3] # Extract rotation part
    P_06 = T_06[:3, 3] # Extract position part

    P_04 = P_06 - JOINT_OFFSETS["D6"] * R_06[:, 2] # Derive wrist center position

    # Calculate q1
    planar_dist = np.hypot(P_04[0], P_04[1])
    phi = np.arcsin(np.clip(JOINT_OFFSETS["D2"] / planar_dist, -1.0, 1.0))
    theta = np.arctan2(P_04[1], P_04[0])
    q1 = [theta - phi, theta + (np.pi + phi)]

    # Calculate q3
    T_01 = T_01_func(q1[0])
    P_04_projected = P_04 - JOINT_OFFSETS["D2"] * T_01[:3, 2]

    r = np.sqrt(P_04_projected[0]**2 + P_04_projected[1]**2)
    s = P_04_projected[2] - JOINT_OFFSETS["D1"]
    D = np.sqrt(r**2 + s**2)

    theta_cos = (D**2 - LINK_LENGTHS["L2"]**2 - JOINT_OFFSETS["D4"]**2) / (2 * LINK_LENGTHS["L2"] * JOINT_OFFSETS["D4"])

    if theta_cos < -1 - EPSILON or theta_cos > 1 + EPSILON:
        return None

    theta_cos = np.clip(theta_cos, -1.0, 1.0)
    theta = np.arccos(theta_cos)
    q3 = [theta, -theta]

     # Calculate q2
    theta = np.arctan2(JOINT_OFFSETS["D4"] * np.sin(q3[0]), LINK_LENGTHS["L2"] + JOINT_OFFSETS["D4"] * np.cos(q3[0]))
    theta_D = (np.pi/2 - np.arctan2(s, r))
    q2 = [theta_D - theta, theta_D + theta]

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

def verify_solutions(T_06, solutions):
    error = []
    for solution in solutions:
        T_06_sol = forward_kinematics(solution)
        error.append(np.linalg.norm(T_06_sol - T_06))
    return error