import numpy as np

EPSILON = 1e-6

LINK_LENGTHS = {
    "L2": 0.2
}

JOINT_OFFSETS = {
    "D1": 0.182,
    "D2": 0.0135,
    "D4": 0.1885,
    "D6": 0.05813392
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