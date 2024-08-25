import numpy as np
from copy import deepcopy
### Setup some constants
# TODO: find best way to handle these constants
DELTA_ACTION_MAGNITUDE_LIMIT = 4.0
DELTA_EPSILON = np.array([1e-7, 1e-7, 1e-7])
DELTA_ACTION_DIRECTION_THRESHOLD = 0.25

SCALE_ACTION_LIMIT = [0.05 * DELTA_ACTION_MAGNITUDE_LIMIT for i in range(3)] + [0.5 * DELTA_ACTION_MAGNITUDE_LIMIT for i in range(3)]

GRIPPER_VELOCITY_THRESHOLD = 0.5
GRIPPER_COMMAND_CHANGE_THRESHOLD = 0.2

REPEAT_LAST_ACTION_TIMES = 10

def demo_obs_to_obs_dict(demo_obs, ind):
    obs_dict = {}
    for o in demo_obs:
        obs_dict[o] = np.array(demo_obs[o][ind])
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

def aggregate_delta_actions(actions, obs=None, **kw_args):
    actions = np.array(actions)
    agg_actions = []
    curr_action = actions[0]

    delta_action_magnitude_limit = kw_args.get('delta_action_magnitude_limit', DELTA_ACTION_MAGNITUDE_LIMIT)
    delta_action_direction_threshold = kw_args.get('delta_action_direction_threshold', DELTA_ACTION_DIRECTION_THRESHOLD)

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

def aggregate_delta_actions_naive(actions, obs=None):
    actions = np.array(actions)
    agg_actions = []
    curr_action = actions[0]

    delta_action_magnitude_limit = kwargs.get('delta_action_magnitude_limit', DELTA_ACTION_MAGNITUDE_LIMIT)


    for i in range(1, actions.shape[0]):
        if sum(np.abs(curr_action[0:3])) > delta_action_magnitude_limit:
            agg_actions.append(curr_action)
            curr_action = actions[i]
        else:
            curr_action[0:6] += actions[i][0:6]
            curr_action[-1] = actions[i][-1]
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

    return agg_actions

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


def create_aggregated_delta_actions_with_gripper_check(actions, obs, **kw_args):

    agg_actions = []
    gripper_vel = obs['robot0_gripper_qvel'][:]*100

    # Define maximum magnitude of delta action
    delta_action_magnitude_limit = kw_args.get('delta_action_magnitude_limit', DELTA_ACTION_MAGNITUDE_LIMIT)

    for i in range(0, actions.shape[0]):
        curr_action = deepcopy(actions[i])
        for j in range(i + 1, actions.shape[0]):
            if sum(np.abs(curr_action[0:3])) > delta_action_magnitude_limit:
                # Magnitude is too large, stop aggregating
                break

            gripper_moving = False
            if max(np.abs(gripper_vel[j])) > GRIPPER_VELOCITY_THRESHOLD:
                gripper_moving = True

            if in_same_direction(actions[j], curr_action) and not gripper_moving:
                # If actions are in the same direction and the gripper is not moving
                curr_action[0:6] += deepcopy(actions[j][0:6])
                curr_action[-1] = deepcopy(actions[j][-1])
            else:
                # Either not in same direction or gripper is moving, stop aggregating
                break

        agg_actions.append(curr_action)
    return np.array(agg_actions)