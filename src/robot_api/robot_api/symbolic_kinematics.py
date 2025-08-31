# symbolic_kinematics.py

import os
import pickle
import hashlib
import json
from sympy import lambdify, symbols, pi, cos, sin,  Matrix
from .config import LINK_LENGTHS, JOINT_OFFSETS

CACHE_DIR = os.path.join(os.path.dirname(__file__), "__cache__")
os.makedirs(CACHE_DIR, exist_ok=True)

params_hash = hashlib.sha256(json.dumps({
    "LINK_LENGTHS": LINK_LENGTHS,
    "JOINT_OFFSETS": JOINT_OFFSETS
}, sort_keys=True).encode()).hexdigest()

CACHE_FILE = os.path.join(CACHE_DIR, f"lambdified_funcs_{params_hash}.pkl")

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

def compute_symbolic_fk():
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

    return {
        "T_01_symbolic": T_01_symbolic,
        "T_02_symbolic": T_02_symbolic,
        "T_03_symbolic": T_03_symbolic,
        "T_04_symbolic": T_04_symbolic,
        "T_05_symbolic": T_05_symbolic,
        "T_06_symbolic": T_06_symbolic
    }

if os.path.exists(CACHE_FILE):
    try:
        with open(CACHE_FILE, "rb") as f:
            symbolic_fk = pickle.load(f)
    except (EOFError, pickle.UnpicklingError):
        symbolic_fk = compute_symbolic_fk()
        with open(CACHE_FILE, "wb") as f:
            pickle.dump(symbolic_fk, f)
else:
    symbolic_fk = compute_symbolic_fk()
    with open(CACHE_FILE, "wb") as f:
        pickle.dump(symbolic_fk, f)

T_01_func = lambdify((theta_1,), symbolic_fk["T_01_symbolic"], modules="numpy")
T_06_func = lambdify(thetas, symbolic_fk["T_06_symbolic"], modules="numpy")
R_03_func = lambdify((theta_1, theta_2, theta_3), symbolic_fk["T_03_symbolic"][:3, :3], modules="numpy")
