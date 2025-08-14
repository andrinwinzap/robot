# symbolic_kinematics.py

from sympy import cos, sin, symbols, pi, Matrix, lambdify, simplify, diff
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

# Homogeneous transformation matrix
HTM_symbolic = Matrix([
    [cos(theta), -sin(theta)*cos(alpha),  sin(theta)*sin(alpha), a*cos(theta)],
    [sin(theta),  cos(theta)*cos(alpha), -cos(theta)*sin(alpha), a*sin(theta)],
    [0, sin(alpha), cos(alpha), d],
    [0, 0, 0, 1]
])

# Symbolic forward kinematics
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

# Jacobian
x = T_06_symbolic[0, 3]
y = T_06_symbolic[1, 3]
z = T_06_symbolic[2, 3]

omega_hat_0_1 = T_01_symbolic[:3, 2]
omega_hat_0_2 = T_02_symbolic[:3, 2]
omega_hat_0_3 = T_03_symbolic[:3, 2]
omega_hat_0_4 = T_04_symbolic[:3, 2]
omega_hat_0_5 = T_05_symbolic[:3, 2]
omega_hat_0_6= T_06_symbolic[:3, 2]

theta_list = [theta_1, theta_2, theta_3, theta_4, theta_5, theta_6]
omega_list = [omega_hat_0_1, omega_hat_0_2, omega_hat_0_3, omega_hat_0_4, omega_hat_0_5, omega_hat_0_6]

cols = []
for i in range(6):
    dx = diff(x, theta_list[i])
    dy = diff(y, theta_list[i])
    dz = diff(z, theta_list[i])
    omega = omega_list[i]
    col = Matrix([dx, dy, dz, omega[0], omega[1], omega[2]])
    cols.append(col)

J_symbolic = simplify(Matrix.hstack(*cols))

# Numerical funcions
T_06_func = lambdify(thetas, T_06_symbolic, modules='numpy')
T_01_func = lambdify((theta_1,), T_01_symbolic, modules="numpy")
R_03_func = lambdify((theta_1, theta_2, theta_3), T_03_symbolic[:3, :3], modules="numpy")
J_func = lambdify(thetas, J_symbolic, modules='numpy')