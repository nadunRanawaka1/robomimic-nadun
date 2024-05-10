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


demo_fn = "/home/nadun/Data/phd_project/robomimic/datasets/lift/ph/all_obs_v141.hdf5"

# demo = nx.nxload(demo_fn)
# print(demo.tree)

### Set video file names
fast_video_fn = "/media/nadun/Data/phd_project/robomimic/videos/lift_sped_up/sped_up_2.mp4"
normal_video_fn = "/media/nadun/Data/phd_project/robomimic/videos/lift_sped_up/normal_2.mp4"


### Read in demo file
demo_file = h5py.File(demo_fn)


### Init env
env_meta = FileUtils.get_env_metadata_from_dataset(demo_fn)
env = EnvUtils.create_env_from_metadata(env_meta,
                                        # render=True,
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


### Make fast video
fast_video_writer = imageio.get_writer(fast_video_fn, fps=20)

counter = 0
fast_reward = 0
time_taken_fast = 0


for ep in demo_file["data"]:
    start = time.time()
    if counter > 10:
        break
    counter += 1

    print(f"processing demo of fast: {counter}")

    demo = demo_file[f'data/{ep}']

    states = demo_file["data/{}/states".format(ep)][()]
    initial_state = dict(states=states[0])
    initial_state["model"] = demo_file["data/{}".format(ep)].attrs["model_file"]

    env.reset()
    env.reset_to(initial_state)

    actions = demo['actions'][:]  # action is [dpos, drot, gripper] where dpos and drot are vectors of size 3
    num_actions = actions.shape[0]
    scaled_steps = num_actions // 2  # count by evens

    # video_img = env.env.sim.render(height=512, width=512, camera_name="agentview")[::-1]
    # fast_video_writer.append_data(video_img)

    for i in range(scaled_steps):
        j = i * 2
        k = j + 1

        first_act = actions[j]
        second_act = actions[k]

        first_act[0:6] += second_act[0:6] #add first and second action to execute at once
        scaled_act = np.copy(first_act)
        scaled_act[-1] = second_act[-1] #keep gripper action from second action

        next_obs, _,_,_ =  env.step(scaled_act)
        # video_img = env.env.sim.render(height=512, width=512, camera_name="agentview")[::-1]
        # fast_video_writer.append_data(video_img)
        # env.render()
        time.sleep(0.05)

    if num_actions % 2:
        act = actions[-1]
        next_obs, _, _, _ = env.step(act)
        # video_img = env.env.sim.render(height=512, width=512, camera_name="agentview")[::-1]
        # fast_video_writer.append_data(video_img)

    time_taken_fast += time.time() - start

    for j in range(num_actions - scaled_steps):
        video_img = env.env.sim.render(height=512, width=512, camera_name="agentview")[::-1]
        fast_video_writer.append_data(video_img)
    fast_reward += env.get_reward()

fast_video_writer.close()

### Make normal video

normal_video_writer = imageio.get_writer(normal_video_fn, fps=20)
counter = 0
normal_reward = 0
time_taken_slow = 0

for ep in demo_file["data"]:
    start = time.time()
    if counter > 10:
        break
    counter += 1

    print(f"processing demo of slow: {counter}")

    demo = demo_file[f'data/{ep}']

    states = demo_file["data/{}/states".format(ep)][()]
    initial_state = dict(states=states[0])
    initial_state["model"] = demo_file["data/{}".format(ep)].attrs["model_file"]

    env.reset()
    env.reset_to(initial_state)

    actions = demo['actions'][:]  # action is [dpos, drot, gripper] where dpos and drot are vectors of size 3

    # video_img = env.env.sim.render(height=512, width=512, camera_name="agentview")[::-1]
    # normal_video_writer.append_data(video_img)

    n = actions.shape[0]

    for i in range(n):
        act = actions[i]

        next_obs, _, _, _ = env.step(act)
        # video_img = env.env.sim.render(height=512, width=512, camera_name="agentview")[::-1]
        # normal_video_writer.append_data(video_img)

        time.sleep(0.05)

    time_taken_slow += time.time() - start

    normal_reward += env.get_reward()

normal_video_writer.close()

print(f"Fast reward: {fast_reward}")
print(f"Normal reward: {normal_reward}")
print(f"Time taken fast: {time_taken_fast}")
print(f"Time take normal: {time_taken_slow}" )