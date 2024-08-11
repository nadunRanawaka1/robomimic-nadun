import numpy as np
import h5py
from copy import deepcopy
from robomimic.dev.dev_utils import in_same_direction
from robomimic.dev.dev_utils import DELTA_ACTION_MAGNITUDE_LIMIT, GRIPPER_CHANGE_THRESHOLD
import nexusformat.nexus as nx






def aggregate_delta_actions_with_gripper_check(actions, obs):

    agg_actions = []
    gripper_obs = obs['robot0_gripper_qpos'][:]*100

    for i in range(0, actions.shape[0]):
        curr_action = deepcopy(actions[i])
        for j in range(i + 1, actions.shape[0]):
            if sum(np.abs(curr_action[0:3])) > DELTA_ACTION_MAGNITUDE_LIMIT:
                # Magnitude is too large, stop aggregating
                break

            curr_gripper_obs = gripper_obs[j]
            prev_gripper_obs = gripper_obs[j - 1]

            gripper_same = True
            if np.sum(np.abs(curr_gripper_obs - prev_gripper_obs)) > GRIPPER_CHANGE_THRESHOLD:
                gripper_same = False

            if in_same_direction(actions[j], curr_action) and gripper_same:
                # If actions are in the same direction and the gripper action does not change, aggregate
                curr_action[0:6] += deepcopy(actions[j][0:6])
                curr_action[-1] = deepcopy(actions[j][-1])
            else:
                # Either not in same direction or gripper action changes, stop aggregating
                break

        agg_actions.append(curr_action)
    return np.array(agg_actions)

demo_fn = "/media/nadun/Data/phd_project/robomimic/datasets/lift/ph/all_obs_v141.hdf5"
processed_fn = "/media/nadun/Data/phd_project/robomimic/datasets/lift/ph/all_obs_v141_aggregated_actions_3.hdf5"

demo_file = nx.nxload(demo_fn)
print(demo_file.tree)

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

    agg_actions = aggregate_delta_actions_with_gripper_check(delta_actions, obs)


    # Write the dataset with new actins
    out_file_ep_grp = out_file_data[f'{ep}']
    del out_file_ep_grp['actions']

    out_file_ep_grp.create_dataset(f"actions", data=agg_actions)
    out_file_ep_grp.attrs['num_samples'] = len(agg_actions)

out_file.close()

