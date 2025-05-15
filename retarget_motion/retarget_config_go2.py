import numpy as np

# Using the actual Go2 URDF
URDF_FILENAME = "../go2_description/urdf/go2_description.urdf"

REF_POS_SCALE = 0.825
INIT_POS = np.array([0, 0, 0.32])
INIT_ROT = np.array([0, 0, 0, 1.0])

# Go2 joint IDs based on actual joint names from the URDF inspection
# Note that the foot joints are FIXED type, but we need them for retargeting
# So we use the corresponding foot joints from the URDF
SIM_TOE_JOINT_IDS = [
    16,  # FR_foot_joint - Front Right foot joint
    34,  # RR_foot_joint - Rear Right foot joint
    7,   # FL_foot_joint - Front Left foot joint
    25,  # RL_foot_joint - Rear Left foot joint
]

# The hip joint IDs for the four legs (all are REVOLUTE type)
SIM_HIP_JOINT_IDS = [
    11,  # FR_hip_joint - Front Right hip joint
    29,  # RR_hip_joint - Rear Right hip joint
    2,   # FL_hip_joint - Front Left hip joint
    20,  # RL_hip_joint - Rear Left hip joint
]

SIM_ROOT_OFFSET = np.array([0, 0, -0.06])

SIM_TOE_OFFSET_LOCAL = [
    np.array([0, -0.05, 0.0]),  # FR
    np.array([0, -0.05, 0.0]),   # FL
    np.array([0, 0.05, 0.0]),  # RR
    np.array([0, 0.05, 0.0])    # RL
]

# Updated default joint pose to match the initial pose for the Go2
# 12 values for the 12 movable joints (3 per leg: hip, thigh, calf)
DEFAULT_JOINT_POSE = np.array([0, 0.5, -1.0, 0, 0.5, -1.0, 0, 0.5, -1.0, 0, 0.5, -1.0])

JOINT_DAMPING = [0.1, 0.05, 0.01,
                 0.1, 0.05, 0.01,
                 0.1, 0.05, 0.01,
                 0.1, 0.05, 0.01]

FORWARD_DIR_OFFSET = np.array([0, 0, 0]) 