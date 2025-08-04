import numpy as np

EPSILON = 1e-6

LINK_LENGTHS = {
    "L2": 200
}

JOINT_OFFSETS = {
    "D1": 182,
    "D2": 13.5,
    "D4": 188.5,
    "D6": 58.13
}

# TODO: Get limits from urdf 
JOINT_LIMITS = [ 
    (-np.pi, np.pi),
    (-np.pi, np.pi),
    (-np.pi, np.pi),
    (-np.pi, np.pi),
    (-np.pi/2, np.pi/2),        
    (-np.pi, np.pi)       
]