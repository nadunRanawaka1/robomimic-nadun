import numpy as np
from copy import deepcopy
import time
import robomimic.utils.tensor_utils as TensorUtils
import robomimic.utils.obs_utils as ObsUtils
from robomimic.envs.env_base import EnvBase
from robomimic.envs.wrappers import EnvWrapper
from robomimic.algo import RolloutPolicy


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

def aggregate_delta_actions_naive(actions, obs=None, **kwargs):
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


def create_aggregated_delta_actions_with_gripper_check(actions, obs, **kwargs):

    agg_actions = []
    gripper_vel = obs['robot0_gripper_qvel'][:]*100

    # Define maximum magnitude of delta action
    delta_action_magnitude_limit = kwargs.get('delta_action_magnitude_limit', DELTA_ACTION_MAGNITUDE_LIMIT)

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


def rollout_open_loop_bc_rnn_joint_actions(policy, env, horizon, render=False, video_writer=None, video_skip=5, return_obs=False, camera_names=None, **kwargs):
    assert isinstance(env, EnvBase) or isinstance(env, EnvWrapper)
    assert isinstance(policy, RolloutPolicy)
    assert not (render and (video_writer is not None))

    policy.start_episode()
    env.reset()
    state_dict = env.get_state()

    # hack that is necessary for robosuite tasks for deterministic action playback
    obs = env.reset_to(state_dict)

    results = {}
    video_count = 0  # video frame counter
    total_reward = 0.
    traj = dict(actions=[], rewards=[], dones=[], states=[], initial_state_dict=state_dict)
    total_inference_time = 0

    start_rollout = time.time()

    rollout_length = policy.policy.algo_config.rnn.horizon
    num_actual_actions = 0

    if return_obs:
        # store observations too
        traj.update(dict(obs=[], next_obs=[]))
    try:
        for step_i in range(horizon):

            # get action from policy
            start = time.time()
            actions = []
            if kwargs["return_action_sequence"]:
                for i in range(rollout_length):
                    act = policy(ob=obs)
                    actions.append(act)
            else:
                act = policy(ob=obs)
                actions.append(act)
            total_inference_time += time.time() - start

            # play regular actions
            for act in actions:
                num_actual_actions += 1
                next_obs, r, done, _ = env.step(act)
                if done:
                    break

            # compute reward
            total_reward += r
            success = env.is_success()["task"]

            # visualization
            if render:
                env.render(mode="human", camera_name=camera_names[0])
            if video_writer is not None:
                if video_count % video_skip == 0:
                    video_img = []
                    for cam_name in camera_names:
                        video_img.append(env.render(mode="rgb_array", height=512, width=512, camera_name=cam_name))
                    video_img = np.concatenate(video_img, axis=1)  # concatenate horizontally
                    video_writer.append_data(video_img)
                video_count += 1

            # collect transition
            traj["actions"].append(act)
            traj["rewards"].append(r)
            traj["dones"].append(done)
            traj["states"].append(state_dict["states"])
            if return_obs:
                # Note: We need to "unprocess" the observations to prepare to write them to dataset.
                #       This includes operations like channel swapping and float to uint8 conversion
                #       for saving disk space.
                traj["obs"].append(ObsUtils.unprocess_obs_dict(obs))
                traj["next_obs"].append(ObsUtils.unprocess_obs_dict(next_obs))

            # break if done or if success
            if done or success:
                break

            # update for next iter
            obs = deepcopy(next_obs)
            state_dict = env.get_state()


    except env.rollout_exceptions as e:
        print("WARNING: got rollout exception {}".format(e))

    total_time_taken = time.time() - start_rollout

    stats = dict(Return=total_reward, Horizon=num_actual_actions, Success_Rate=float(success), Time_Taken_in_rollout= total_time_taken)

    if return_obs:
        # convert list of dict to dict of list for obs dictionaries (for convenient writes to hdf5 dataset)
        traj["obs"] = TensorUtils.list_of_flat_dict_to_dict_of_list(traj["obs"])
        traj["next_obs"] = TensorUtils.list_of_flat_dict_to_dict_of_list(traj["next_obs"])

    # list to numpy array
    for k in traj:
        if k == "initial_state_dict":
            continue
        if isinstance(traj[k], dict):
            for kp in traj[k]:
                traj[k][kp] = np.array(traj[k][kp])
        else:
            traj[k] = np.array(traj[k])

    return stats, traj