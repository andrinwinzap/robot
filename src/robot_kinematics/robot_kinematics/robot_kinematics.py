# robot_kinematics.py

import numpy as np
from sympy import cos, sin, symbols, pi, Matrix, lambdify

theta, d, alpha, a = symbols('theta d alpha a')

theta_1, theta_2, theta_3, theta_4, theta_5, theta_6 = symbols('theta_1 theta_2 theta_3 theta_4 theta_5 theta_6')
thetas = [theta_1, theta_2, theta_3, theta_4, theta_5, theta_6]

EPSILON = 1e-6

D1 = 182
D2 = 13.5
D6 = 58.13
L2 = 200
D4 = 188.5

JOINT_LIMITS = [
    (-np.pi, np.pi),
    (-np.pi, np.pi),
    (-np.pi, np.pi),
    (-np.pi, np.pi),
    (-np.pi/2, np.pi/2),        
    (-np.pi, np.pi)       
]

DH_Params = [
    {theta: theta_1, d: D1, alpha: pi/2, a: 0},
    {theta: theta_2 + (pi/2), d: D2, alpha: 0, a: L2},
    {theta: theta_3 - (pi/2), d: 0, alpha: -pi/2, a: 0},
    {theta: theta_4, d: D4, alpha: -pi/2, a: 0},
    {theta: theta_5, d: 0, alpha: pi/2, a: 0},
    {theta: theta_6, d: D6, alpha: 0, a: 0}
]

HTM_symbolic = Matrix([
    [cos(theta), -sin(theta)*cos(alpha),  sin(theta)*sin(alpha), a*cos(theta)],
    [sin(theta),  cos(theta)*cos(alpha), -cos(theta)*sin(alpha), a*sin(theta)],
    [0, sin(alpha), cos(alpha), d],
    [0, 0, 0, 1]
])

T_01_symbolic = HTM_symbolic.subs(DH_Params[0])
T_12_symbolic = HTM_symbolic.subs(DH_Params[1])
T_23_symbolic = HTM_symbolic.subs(DH_Params[2])
T_34_symbolic = HTM_symbolic.subs(DH_Params[3])
T_45_symbolic = HTM_symbolic.subs(DH_Params[4])
T_56_symbolic = HTM_symbolic.subs(DH_Params[5])

T_02_symbolic = T_01_symbolic * T_12_symbolic
T_03_symbolic = T_02_symbolic * T_23_symbolic
T_04_symbolic = T_03_symbolic * T_34_symbolic
T_05_symbolic = T_04_symbolic * T_45_symbolic
T_06_symbolic = T_05_symbolic * T_56_symbolic

T_36_symbolic = T_34_symbolic * T_45_symbolic * T_56_symbolic

T_06_func = lambdify(thetas, T_06_symbolic, modules='numpy')

T_01_func = lambdify((theta_1,), T_01_symbolic, modules="numpy")
R_03_func = lambdify((theta_1, theta_2, theta_3), T_03_symbolic[:3, :3], modules="numpy")

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

    P_04 = P_06 - D6 * R_06[:, 2] # Derive wrist center position

    # Calculate q1
    planar_dist = np.hypot(P_04[0], P_04[1])
    phi = np.arcsin(np.clip(D2 / planar_dist, -1.0, 1.0))
    theta = np.arctan2(P_04[1], P_04[0])
    q1 = [theta + (np.pi - phi), theta + phi]
    
    # Calculate q3
    T_01 = T_01_func(q1[0])
    P_04_projected = P_04 - D2 * T_01[:3, 2]

    r = np.sqrt(P_04_projected[0]**2 + P_04_projected[1]**2)
    s = P_04_projected[2] - D1
    D = np.sqrt(r**2 + s**2)

    theta_cos = (D**2 - L2**2 - D4**2) / (2 * L2 * D4)

    if theta_cos < -1 - EPSILON or theta_cos > 1 + EPSILON:
        return None

    theta_cos = np.clip(theta_cos, -1.0, 1.0)
    theta = np.arccos(theta_cos)
    q3 = [theta, -theta]

     # Calculate q2
    theta = np.arctan2(D4 * np.sin(q3[0]), L2 + D4 * np.cos(q3[0]))
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

        for solution in T_36_solutions:
            if check_limits(solution):
                q4,q5,q6 = solution
                T_06_solutions.append([q1, q2, q3, q4, q5, q6])
            else:
                print(f"not within limits: {solution}")

    return T_06_solutions

def check_solutions(T_06, solutions):
    for solution in solutions:
        T_06_sol = forward_kinematics(solution)
        print(f"T_06 error: {np.linalg.norm(T_06_sol - T_06)}")