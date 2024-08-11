
### Setup some constants
DELTA_ACTION_MAGNITUDE_LIMIT = 3.0
DELTA_EPSILON = np.array([1e-7, 1e-7, 1e-7])
DELTA_ACTION_DIRECTION_THRESHOLD = 0.25
# SCALE_ACTION_LIMIT_MIN = [-0.05, -0.05, -0.05, -0.5, -0.5, -0.5]
# SCALE_ACTION_LIMIT_MAX = [0.05, 0.05, 0.05, 0.5, 0.5, 0.5]

SCALE_ACTION_LIMIT = 0.15



def in_same_direction(act1, act2):
    act1_delta_pos = act1[0:3] + DELTA_EPSILON
    # First normalize both
    act1_norm = np.linalg.norm(act1_delta_pos)
    act1_delta_pos /= act1_norm

    act2_delta_pos = act2[0:3] + DELTA_EPSILON
    act2_norm = np.linalg.norm(act2_delta_pos)
    act2_delta_pos /= act2_norm

    # Then use dot product to check
    d_prod = np.dot(act1_delta_pos, act2_delta_pos)

    if d_prod < DELTA_ACTION_DIRECTION_THRESHOLD:
        return False
    else:
        return True