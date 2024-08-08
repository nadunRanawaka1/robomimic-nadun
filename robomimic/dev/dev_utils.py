import numpy as np

### Setup some constants
# TODO: find best way to handle these constants
DELTA_ACTION_MAGNITUDE_LIMIT = 1.0
DELTA_EPSILON = np.array([1e-7, 1e-7, 1e-7])
DELTA_ACTION_DIRECTION_THRESHOLD = 0.25
SCALE_ACTION_LIMIT = [0.05, 0.05, 0.05, 0.5, 0.5, 0.5]




def demo_obs_to_obs_dict(demo_obs, ind):
    obs_dict = {}
    for o in demo_obs:
        obs_dict[o] = demo_obs[o][ind]
    return obs_dict

def aggregate_delta_actions(actions):
    actions = np.array(actions)
    agg_actions = []
    curr_action = actions[0]

    for i in range(1, actions.shape[0]):
        if sum(np.abs(curr_action[0:3])) > DELTA_ACTION_MAGNITUDE_LIMIT:
            agg_actions.append(curr_action)
            curr_action = actions[i]
            continue

        ### check if current action and next action are in similar directions
        next_action_delta_pos = actions[i][0:3] + DELTA_EPSILON
        # First normalize both
        next_action_norm = np.linalg.norm(next_action_delta_pos)
        next_action_delta_pos /= next_action_norm
        curr_action_delta_pos = np.copy(curr_action[0:3]) + DELTA_EPSILON
        curr_action_norm = np.linalg.norm(curr_action_delta_pos)
        curr_action_delta_pos /= curr_action_norm
        # Then use dot product to check
        d_prod = np.dot(next_action_delta_pos, curr_action_delta_pos)

        if d_prod < DELTA_ACTION_DIRECTION_THRESHOLD:  # curr action and next action are not in the same direction
            agg_actions.append(curr_action)
            curr_action = actions[i]
        else:
            curr_action[0:6] += actions[i][0:6]
            curr_action[-1] = actions[i][-1]

    agg_actions.append(curr_action)
    return agg_actions