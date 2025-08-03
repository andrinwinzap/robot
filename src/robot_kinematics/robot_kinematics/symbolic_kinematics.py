# symbolic_kinematics.py

from sympy import cos, sin, symbols, pi, Matrix, lambdify
from .config import LINK_LENGTHS, JOINT_OFFSETS

theta, d, alpha, a = symbols('theta d alpha a')

theta_1, theta_2, theta_3, theta_4, theta_5, theta_6 = symbols('theta_1 theta_2 theta_3 theta_4 theta_5 theta_6')
thetas = [theta_1, theta_2, theta_3, theta_4, theta_5, theta_6]

DH_Params = [
    {theta: theta_1, d: JOINT_OFFSETS["D1"], alpha: -pi/2, a: 0},
    {theta: theta_2 - (pi/2), d: JOINT_OFFSETS["D2"], alpha: 0, a: LINK_LENGTHS["L2"]},
    {theta: theta_3 + (pi/2), d: 0, alpha: pi/2, a: 0},
    {theta: theta_4, d: JOINT_OFFSETS["D4"], alpha: -pi/2, a: 0},
    {theta: theta_5, d: 0, alpha: pi/2, a: 0},
    {theta: theta_6, d: JOINT_OFFSETS["D6"], alpha: 0, a: 0}
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
