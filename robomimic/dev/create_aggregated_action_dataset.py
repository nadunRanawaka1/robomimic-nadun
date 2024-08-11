import numpy as np
import h5py
from copy import deepcopy



### Setup some constants
DELTA_ACTION_MAGNITUDE_LIMIT = 2.0
DELTA_EPSILON = np.array([1e-7, 1e-7, 1e-7])
DELTA_ACTION_DIRECTION_THRESHOLD = 0.25
# SCALE_ACTION_LIMIT_MIN = [-0.05, -0.05, -0.05, -0.5, -0.5, -0.5]
# SCALE_ACTION_LIMIT_MAX = [0.05, 0.05, 0.05, 0.5, 0.5, 0.5]

SCALE_ACTION_LIMIT = 0.5

GRIPPER_CHANGE_THRESHOLD = 0.3




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

def aggregate_delta_actions_with_gripper_check(actions, obs):

    agg_actions = []
    ret_obs_ind = [0]
    curr_action = actions[0]


    gripper_obs = obs['robot0_gripper_qpos'][:]*100

    for i in range(1, actions.shape[0]):
        if sum(np.abs(curr_action[0:3])) > DELTA_ACTION_MAGNITUDE_LIMIT:
            agg_actions.append(curr_action)
            curr_action = deepcopy(actions[i])
            ret_obs_ind.append(i)
            continue

        curr_gripper_obs = gripper_obs[i]
        prev_gripper_obs = gripper_obs[i - 1]

        gripper_same = True
        if np.sum(np.abs(curr_gripper_obs - prev_gripper_obs)) > GRIPPER_CHANGE_THRESHOLD:
            gripper_same = False

        if in_same_direction(actions[i], curr_action) and gripper_same:
            # If actions are in the same direction and the gripper action does not change, aggregate
            curr_action[0:6] += deepcopy(actions[i][0:6])
            curr_action[-1] = deepcopy(actions[i][-1])
        else:
            # Either not in same direction or gripper action changes
            agg_actions.append(curr_action)
            curr_action = deepcopy(actions[i])
            ret_obs_ind.append(i)
    agg_actions.append(curr_action)
    return agg_actions, ret_obs_ind

demo_fn = "/media/nadun/Data/phd_project/robomimic/datasets/lift/ph/all_obs_v141.hdf5"
processed_fn = "/media/nadun/Data/phd_project/robomimic/datasets/lift/ph/all_obs_v141_aggregated_actions_2.hdf5"

demo_file = h5py.File(demo_fn)
out_file  = h5py.File(processed_fn, 'w')

for obj in demo_file.keys():
    demo_file.copy(obj, out_file)

out_file_data = out_file['data']

for ep in demo_file["data"]:
    demo = demo_file[f'data/{ep}']
    delta_actions = demo["actions"][:]
    obs = demo['obs']
    print(f"processing demo {ep}")

    agg_actions, obs_ind = aggregate_delta_actions_with_gripper_check(delta_actions, obs)
    processed_obs_dict = {}
    for o in obs:
        processed_obs = []
        orig = obs[o][:]
        for i in obs_ind:
            processed_obs.append(orig[i])
        processed_obs_dict[o] = np.array(processed_obs)

    # Write the dataset with new actions and observations
    out_file_ep_grp = out_file_data[f'{ep}']
    del out_file_ep_grp['actions']
    del out_file_ep_grp['obs']
    for o in obs:
        out_file_ep_grp.create_dataset(f"obs/{o}", data=np.array(processed_obs_dict[o]))
    out_file_ep_grp.create_dataset(f"actions", data=agg_actions)
    out_file_ep_grp.attrs['num_samples'] = len(agg_actions)

out_file.close()

