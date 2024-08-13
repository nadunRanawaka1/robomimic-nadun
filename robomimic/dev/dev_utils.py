import numpy as np

### Setup some constants
# TODO: find best way to handle these constants
DELTA_ACTION_MAGNITUDE_LIMIT = 3.0
DELTA_EPSILON = np.array([1e-7, 1e-7, 1e-7])
DELTA_ACTION_DIRECTION_THRESHOLD = 0.25
SCALE_ACTION_LIMIT = [0.05, 0.05, 0.05, 0.5, 0.5, 0.5]

GRIPPER_CHANGE_THRESHOLD = 0.3
GRIPPER_COMMAND_CHANGE_THRESHOLD = 0.2

def demo_obs_to_obs_dict(demo_obs, ind):
    obs_dict = {}
    for o in demo_obs:
        obs_dict[o] = demo_obs[o][ind]
    return obs_dict

def in_same_direction(act1, act2, threshold=DELTA_ACTION_DIRECTION_THRESHOLD):
    """
    Check if two actions are in the same direction
    """
    act1_delta_pos = act1[0:3] + DELTA_EPSILON
    act1_norm = np.linalg.norm(act1_delta_pos)
    act1_delta_pos /= act1_norm

    act2_delta_pos = act2[0:3] + DELTA_EPSILON
    act2_norm = np.linalg.norm(act2_delta_pos)
    act2_delta_pos /= act2_norm

    d_prod_1 = np.dot(act1_delta_pos, act2_delta_pos)

    act1_delta_angle = act1[3:6] + DELTA_EPSILON
    act1_norm = np.linalg.norm(act1_delta_angle)
    act1_delta_angle /= act1_norm

    act2_delta_angle = act2[3:6] + DELTA_EPSILON
    act2_norm = np.linalg.norm(act2_delta_angle)
    act2_delta_angle /= act2_norm

    d_prod_2 = np.dot(act1_delta_angle, act2_delta_angle)

    if d_prod_1 > threshold and d_prod_2 > threshold:
        return True
    else:
        return False

def aggregate_delta_actions(actions, obs=None, **kwargs):
    actions = np.array(actions)
    agg_actions = []
    curr_action = actions[0]

    delta_action_magnitude_limit = kwargs.get('delta_action_magnitude_limit', DELTA_ACTION_MAGNITUDE_LIMIT)
    delta_action_direction_threshold = kwargs.get('delta_action_direction_threshold', DELTA_ACTION_DIRECTION_THRESHOLD)

    for i in range(1, actions.shape[0]):
        if sum(np.abs(curr_action[0:3])) > delta_action_magnitude_limit:
            agg_actions.append(curr_action)
            curr_action = actions[i]
            continue

        # ### check if current action and next action are in similar directions
        # next_action_delta_pos = actions[i][0:3] + DELTA_EPSILON
        # # First normalize both
        # next_action_norm = np.linalg.norm(next_action_delta_pos)
        # next_action_delta_pos /= next_action_norm
        # curr_action_delta_pos = np.copy(curr_action[0:3]) + DELTA_EPSILON
        # curr_action_norm = np.linalg.norm(curr_action_delta_pos)
        # curr_action_delta_pos /= curr_action_norm
        # # Then use dot product to check
        # d_prod = np.dot(next_action_delta_pos, curr_action_delta_pos)

        same_dir = in_same_direction(np.copy(actions[i]), np.copy(curr_action), delta_action_direction_threshold)

        if same_dir:
            # If the current aggregated action and next camera are in the same direction, keep aggregating
            curr_action[0:6] += actions[i][0:6]
            curr_action[-1] = actions[i][-1]
        else:
            # curr action and next action are not in the same direction
            # append the current action and start aggregating new
            agg_actions.append(curr_action)
            curr_action = actions[i]

    agg_actions.append(curr_action)
    return agg_actions


def aggregate_delta_actions_with_gripper_check(actions, gripper_obs):
    actions = np.array(actions)
    agg_actions = []
    curr_action = actions[0]


    for i in range(1, actions.shape[0]):
        if sum(np.abs(curr_action[0:3])) > DELTA_ACTION_MAGNITUDE_LIMIT:
            agg_actions.append(curr_action)
            curr_action = actions[i]
            continue

        curr_gripper_obs = gripper_obs[i]
        prev_gripper_obs = gripper_obs[i-1]

        gripper_same = True
        if np.sum(np.abs(curr_gripper_obs - prev_gripper_obs)) > GRIPPER_CHANGE_THRESHOLD:
            gripper_same = False

        if in_same_direction(actions[i], curr_action) and gripper_same:
            # If actions are in the same direction and the gripper action does not change, aggregate
            curr_action[0:6] += actions[i][0:6]
            curr_action[-1] = actions[i][-1]
        else:
            # Either not in same direction or gripper action changes
            agg_actions.append(curr_action)
            curr_action = actions[i]

    ObsUtils.initialize_obs_utils_with_obs_specs(obs_modality_spec)
    return env, demo_file

def gripper_command_changed(gripper_traj, last_gripper_act):
    """
    Args:
        gripper_traj: nd.Array of gripper actions (1, N)
        last_gripper_act: single gripper action - float

    Made with help from ChatGPT
    """
    # Compute the pairwise absolute differences between gripper commands in the trajectory
    diff_matrix = np.abs(gripper_traj[:, np.newaxis] - gripper_traj)

    # Extract the upper triangle of the difference matrix (excluding the diagonal)
    upper_triangle_diff = np.triu(diff_matrix, k=1)

    # Check if any difference exceeds the threshold
    changed = np.any(upper_triangle_diff > GRIPPER_COMMAND_CHANGE_THRESHOLD)

    if changed or np.abs((gripper_traj[0]) - last_gripper_act) > GRIPPER_COMMAND_CHANGE_THRESHOLD:
        return True
    else:
        return False
