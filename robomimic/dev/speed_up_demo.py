import os
import json
import h5py
import argparse
import numpy as np
from copy import deepcopy
from tqdm import tqdm
import time
import imageio

import robomimic.utils.tensor_utils as TensorUtils
import robomimic.utils.file_utils as FileUtils
import robomimic.utils.env_utils as EnvUtils
import robomimic.utils.obs_utils as ObsUtils
from robomimic.envs.env_base import EnvBase
import nexusformat.nexus as nx

### Setup some constants
DELTA_ACTION__MAGNITUDE_LIMIT = 1.0
DELTA_EPSILON = np.array([1e-7, 1e-7, 1e-7])
DELTA_ACTION_DIRECTION_THRESHOLD = 0.25



demo_fn = "/home/nadun/Data/phd_project/robomimic/datasets/lift/ph/all_obs_v141.hdf5"

# demo = nx.nxload(demo_fn)
# print(demo.tree)

### Set video file names
fast_video_fn = "/media/nadun/Data/phd_project/robomimic/videos/lift_sped_up/sped_up_3.mp4"
normal_video_fn = "/media/nadun/Data/phd_project/robomimic/videos/lift_sped_up/normal_3.mp4"


### Read in demo file
demo_file = h5py.File(demo_fn)


### Init env
env_meta = FileUtils.get_env_metadata_from_dataset(demo_fn)
env = EnvUtils.create_env_from_metadata(env_meta,
                                        render=True,
                                        use_image_obs=True)

### Initializing OBS specs
demo = demo_file['data/demo_0']

obs = demo['obs']
obs_modality_spec = {"obs": {
    "low_dim": [],
    "rgb": []
    }
}
for obs_key in obs:
    if "image" in obs_key:
        obs_modality_spec["obs"]["rgb"].append(obs_key)
    else:
        obs_modality_spec["obs"]["low_dim"].append(obs_key)

ObsUtils.initialize_obs_utils_with_obs_specs(obs_modality_spec)

actions = demo["actions"][:]
print()

def aggregate_delta_actions(demo):
    actions = demo["actions"][:]

    agg_actions = []
    curr_action = actions[0]

    for i in range(1, actions.shape[0]):
        if sum(np.abs(curr_action[0:3])) > DELTA_ACTION__MAGNITUDE_LIMIT:
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

        if d_prod < DELTA_ACTION_DIRECTION_THRESHOLD: # curr action and next action are not in the same direction
            agg_actions.append(curr_action)
            curr_action = actions[i]
        else:
            curr_action[0:6] += actions[i][0:6]
            curr_action[-1] = actions[i][-1]

    agg_actions.append(curr_action)
    return agg_actions


def replay_by_aggregating(demo_file, limit, video_fn=None):

    print("Replaying by aggregating")

    if video_fn is not None:
        video_writer = imageio.get_writer(video_fn, fps=20)

    counter = 0
    reward = 0
    time_taken = 0
    num_agg_actions = 0

    for ep in demo_file["data"]:

        if counter > limit:
            break
        counter += 1

        # print(f"Replaying demo : {counter}")

        demo = demo_file[f'data/{ep}']


        states = demo_file["data/{}/states".format(ep)][()]
        initial_state = dict(states=states[0])
        initial_state["model"] = demo_file["data/{}".format(ep)].attrs["model_file"]

        env.reset()
        env.reset_to(initial_state)

        agg_actions = aggregate_delta_actions(demo)  # action is [dpos, drot, gripper] where dpos and drot are vectors of size 3
        num_agg_actions += len(agg_actions)

        start = time.time()
        if video_fn is not None:
            video_img = env.env.sim.render(height=512, width=512, camera_name="agentview")[::-1]
            video_writer.append_data(video_img)

        for act in agg_actions:

            next_obs, _, _, _ = env.step(act)
            if video_fn is not None:
                video_img = env.env.sim.render(height=512, width=512, camera_name="agentview")[::-1]
                video_writer.append_data(video_img)
            # env.render()
            # time.sleep(0.05)

        time_taken += time.time() - start
        reward += env.get_reward()

        num_normal_actions = demo["actions"].shape[0]
        ### We do this to synchronize the normal video and aggregated one
        for j in range(num_normal_actions - len(agg_actions)):
            if video_fn is not None:
                video_img = env.env.sim.render(height=512, width=512, camera_name="agentview")[::-1]
                video_writer.append_data(video_img)


    if video_fn is not None:
        video_writer.close()

    print(f"Reward by aggregating: {reward}")
    print(f"Time taken aggregating: {time_taken}")
    print(f"Num agg actions: {num_agg_actions}")

def replay_by_skipping(demo_file, limit, video_fn=None):
    ### Make fast video

    if video_fn is not None:
        video_writer = imageio.get_writer(video_fn, fps=20)

    counter = 0
    fast_reward = 0
    time_taken_fast = 0
    num_skipped_actions = 0

    for ep in demo_file["data"]:

        if counter > limit:
            break
        counter += 1

        # print(f"processing demo of fast: {counter}")

        demo = demo_file[f'data/{ep}']

        states = demo_file["data/{}/states".format(ep)][()]
        initial_state = dict(states=states[0])
        initial_state["model"] = demo_file["data/{}".format(ep)].attrs["model_file"]

        env.reset()
        env.reset_to(initial_state)

        actions = demo['actions'][:]  # action is [dpos, drot, gripper] where dpos and drot are vectors of size 3
        num_actions = actions.shape[0]
        scaled_steps = num_actions // 2  # count by evens
        num_skipped_actions += scaled_steps
        start = time.time()

        if video_fn is not None:
            video_img = env.env.sim.render(height=512, width=512, camera_name="agentview")[::-1]
            video_writer.append_data(video_img)

        for i in range(scaled_steps):
            j = i * 2
            k = j + 1

            first_act = actions[j]
            second_act = actions[k]

            first_act[0:6] += second_act[0:6] #add first and second action to execute at once
            scaled_act = np.copy(first_act)
            scaled_act[-1] = second_act[-1] #keep gripper action from second action

            next_obs, _,_,_ =  env.step(scaled_act)
            if video_fn is not None:
                video_img = env.env.sim.render(height=512, width=512, camera_name="agentview")[::-1]
                video_writer.append_data(video_img)
            # env.render()
            # time.sleep(0.05)

        if num_actions % 2:
            act = actions[-1]
            next_obs, _, _, _ = env.step(act)
            if video_fn is not None:
                video_img = env.env.sim.render(height=512, width=512, camera_name="agentview")[::-1]
                video_writer.append_data(video_img)
            num_skipped_actions += 1

        time_taken_fast += time.time() - start

        if video_fn is not None:
            video_img = env.env.sim.render(height=512, width=512, camera_name="agentview")[::-1]
            video_writer.append_data(video_img)
        fast_reward += env.get_reward()

    if video_fn is not None:
        video_writer.close()

    print(f"Skip reward: {fast_reward}")
    print(f"Time taken skipping: {time_taken_fast}")
    print(f"Num skipped actions:{num_skipped_actions}")


def replay_normal_speed(demo_file, limit, video_fn=None):

    if video_fn is not None:
        video_writer = imageio.get_writer(video_fn, fps=20)

    counter = 0
    normal_reward = 0
    time_taken = 0
    num_actions = 0

    for ep in demo_file["data"]:

        if counter > limit:
            break
        counter += 1

        # print(f"processing demo of slow: {counter}")

        demo = demo_file[f'data/{ep}']

        states = demo_file["data/{}/states".format(ep)][()]
        initial_state = dict(states=states[0])
        initial_state["model"] = demo_file["data/{}".format(ep)].attrs["model_file"]

        env.reset()
        env.reset_to(initial_state)

        actions = demo['actions'][:]  # action is [dpos, drot, gripper] where dpos and drot are vectors of size 3
        start = time.time()

        if video_fn is not None:
            video_img = env.env.sim.render(height=512, width=512, camera_name="agentview")[::-1]
            video_writer.append_data(video_img)

        n = actions.shape[0]
        num_actions += n

        for i in range(n):
            act = actions[i]

            next_obs, _, _, _ = env.step(act)
            # env.render()
            if video_fn is not None:
                video_img = env.env.sim.render(height=512, width=512, camera_name="agentview")[::-1]
                video_writer.append_data(video_img)

            # time.sleep(0.05)

        time_taken += time.time() - start

        normal_reward += env.get_reward()

    if video_fn is not None:
        video_writer.close()
    print(f"Normal reward: {normal_reward}")
    print(f"Time take normal: {time_taken}")
    print(f"Num normal actions: {num_actions}")



### execute functions
replay_by_aggregating(demo_file, 10, video_fn="/media/nadun/Data/phd_project/robomimic/videos/lift_sped_up/aggregated_actions.mp4")
replay_by_skipping(demo_file, 10, video_fn="/media/nadun/Data/phd_project/robomimic/videos/lift_sped_up/skipping_actions.mp4")
replay_normal_speed(demo_file, 10, video_fn="/media/nadun/Data/phd_project/robomimic/videos/lift_sped_up/normal.mp4")





