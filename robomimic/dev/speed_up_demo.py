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
import robomimic.utils.transform_utils as TransformUtils
from robomimic.envs.env_base import EnvBase
from robomimic.dev.dev_utils import aggregate_delta_actions



import nexusformat.nexus as nx


### END IMPORTS

### Setup some constants
DELTA_ACTION_MAGNITUDE_LIMIT = 3.0
DELTA_EPSILON = np.array([1e-7, 1e-7, 1e-7])
DELTA_ACTION_DIRECTION_THRESHOLD = 0.25
# SCALE_ACTION_LIMIT_MIN = [-0.05, -0.05, -0.05, -0.5, -0.5, -0.5]
# SCALE_ACTION_LIMIT_MAX = [0.05, 0.05, 0.05, 0.5, 0.5, 0.5]

SCALE_ACTION_LIMIT = 0.5

GRIPPER_CHANGE_THRESHOLD = 0.3



demo_fn = "/media/nadun/Data/phd_project/robomimic/datasets/lift/ph/all_obs_v141.hdf5"
#
# demo = nx.nxload(demo_fn)
# print(demo.tree)

### Set video file names
fast_video_fn = "/media/nadun/Data/phd_project/robomimic/videos/lift_sped_up/sped_up_3.mp4"
normal_video_fn = "/media/nadun/Data/phd_project/robomimic/videos/lift_sped_up/normal_3.mp4"


### Read in demo file

def complete_setup_for_replay(demo_fn):
    demo_file = h5py.File(demo_fn)

    ### Init env
    env_meta = FileUtils.get_env_metadata_from_dataset(demo_fn)

    ### Resetting controller max control input and output here
    # env_meta['env_kwargs']['controller_configs']['input_min'] = -DELTA_ACTION_MAGNITUDE_LIMIT
    # env_meta['env_kwargs']['controller_configs']['input_max'] = DELTA_ACTION_MAGNITUDE_LIMIT
    #
    # env_meta['env_kwargs']['controller_configs']['output_min'] = -SCALE_ACTION_LIMIT
    # env_meta['env_kwargs']['controller_configs']['output_max'] = SCALE_ACTION_LIMIT
    env_meta['env_kwargs']['controller_configs']['control_delta'] = False

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

}


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


def aggregate_delta_actions(actions, gripper_obs=None):

    actions = np.array(actions)
    agg_actions = []
    curr_action = actions[0]


    for i in range(1, actions.shape[0]):

        # If magnitude of current aggregated action greater than limit, add to list and start new aggregation
        if sum(np.abs(curr_action[0:3])) > DELTA_ACTION_MAGNITUDE_LIMIT:
            agg_actions.append(curr_action)
            curr_action = actions[i]
            continue

        if in_same_direction(actions[i], curr_action):
            # If actions are in the same direction, aggregate
            curr_action[0:6] += actions[i][0:6]
            curr_action[-1] = actions[i][-1]
        else:
            # Add current aggregated action to list and start new aggregation
            agg_actions.append(curr_action)
            curr_action = actions[i]


        ### check if current action and next action are in similar directions
        # next_action_delta_pos = actions[i][0:3] + DELTA_EPSILON
        # # First normalize both
        # next_action_norm = np.linalg.norm(next_action_delta_pos)
        # next_action_delta_pos /= next_action_norm
        # curr_action_delta_pos = np.copy(curr_action[0:3]) + DELTA_EPSILON
        # curr_action_norm = np.linalg.norm(curr_action_delta_pos)
        # curr_action_delta_pos /= curr_action_norm
        # # Then use dot product to check
        # d_prod = np.dot(next_action_delta_pos, curr_action_delta_pos)
        #
        # if d_prod < DELTA_ACTION_DIRECTION_THRESHOLD: # curr action and next action are not in the same direction
        #     agg_actions.append(curr_action)
        #     curr_action = actions[i]
        # else:
        #     curr_action[0:6] += actions[i][0:6]
        #     curr_action[-1] = actions[i][-1]



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

def replay_by_aggregating(demo_fn, limit, video_fn=None):

    env, demo_file = complete_setup_for_replay(demo_fn)


def replay_by_aggregating(demo_file, limit, aggregating_function=aggregate_delta_actions, video_fn=None):


    if video_fn is not None:
        video_writer = imageio.get_writer(video_fn, fps=20)

    counter = 0
    reward = 0
    time_taken = 0
    num_agg_actions = 0

    for ep in demo_file["data"]:

        counter += 1
        if counter > limit:
            break

        if counter % 20 == 0 :
            print(f"Replaying demo : {counter}")

        demo = demo_file[f'data/{ep}']
        delta_actions = demo["actions"][:]


        gripper_obs = demo['obs/robot0_gripper_qpos'][:]*100

        states = demo_file["data/{}/states".format(ep)][()]
        initial_state = dict(states=states[0])
        initial_state["model"] = demo_file["data/{}".format(ep)].attrs["model_file"]

        env.reset()
        env.reset_to(initial_state)

        agg_actions = aggregating_function(demo["actions"][:], gripper_obs)  # action is [dpos, drot, gripper] where dpos and drot are vectors of size 3

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

def replay_by_skipping(demo_fn, limit, video_fn=None):
    ### Make fast video

    env, demo_file = complete_setup_for_replay(demo_fn)

    if video_fn is not None:
        video_writer = imageio.get_writer(video_fn, fps=20)

    counter = 0
    fast_reward = 0
    time_taken_fast = 0
    num_skipped_actions = 0

    for ep in demo_file["data"]:

        counter += 1

        if counter > limit:
            break

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


def replay_normal_speed(demo_fn, limit, video_fn=None):
    
    print("Replaying normal")
    env, demo_file = complete_setup_for_replay(demo_fn)


    if video_fn is not None:
        video_writer = imageio.get_writer(video_fn, fps=20)

    counter = 0
    normal_reward = 0
    time_taken = 0
    num_actions = 0

    for ep in demo_file["data"]:
        counter += 1
        if counter > limit:
            break

        if counter % 20 == 0 :
            print(f"Replaying demo : {counter}")


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

            if video_fn is not None:
                video_img = env.env.sim.render(height=512, width=512, camera_name="agentview")[::-1]
                video_writer.append_data(video_img)\

        time_taken += time.time() - start

        normal_reward += env.get_reward()

    if video_fn is not None:
        video_writer.close()
    print(f"Normal reward: {normal_reward}")
    print(f"Time taken normal: {time_taken}")
    print(f"Num normal actions: {num_actions}")


def replay_joint_position_actions(demo_fn, limit, video_fn):

    env, demo_file = complete_setup_for_replay(demo_fn)

    ### Init env
    env_meta = FileUtils.get_env_metadata_from_dataset(demo_fn)
    joint_controller_fp = "/media/nadun/Data/phd_project/robosuite/robosuite/controllers/config/joint_position_nadun.json"
    controller_configs = json.load(open(joint_controller_fp))
    env_meta["env_kwargs"]["controller_configs"] = controller_configs

    env = EnvUtils.create_env_from_metadata(env_meta,
                                            render=True,
                                            use_image_obs=True)

    if video_fn is not None:
        video_writer = imageio.get_writer(video_fn, fps=20)

    counter = 0
    normal_reward = 0
    time_taken = 0
    num_actions = 0

    for ep in demo_file["data"]:
        counter += 1
        if counter > limit:
            break

        print(f"processing demo: {counter}")

        demo = demo_file[f'data/{ep}']

        states = demo_file["data/{}/states".format(ep)][()]
        initial_state = dict(states=states[0])
        initial_state["model"] = demo_file["data/{}".format(ep)].attrs["model_file"]

        env.reset()
        env.reset_to(initial_state)

        actions = demo['obs/robot0_joint_pos'][:]  # action is [joint_pos, gripper] where dpos and drot are vectors of size 3
        actions = demo['obs/robot0_joint_vel'][:]
        gripper_actions = demo['actions'][:,-1]
        gripper_actions = np.expand_dims(gripper_actions, axis=1)

        actions = np.concatenate([actions, gripper_actions], axis=1)

        start = time.time()

        if video_fn is not None:
            video_img = env.env.sim.render(height=512, width=512, camera_name="agentview")[::-1]
            video_writer.append_data(video_img)

        n = actions.shape[0]
        num_actions += n

        joint_pos_list = []
        for i in range(n):
            act = np.copy(actions[i])
            next_obs = env.get_observation()
            joint_pos = next_obs["robot0_joint_pos"]

            joint_pos_list.append(joint_pos)

            act[:-1] = act[:-1] - joint_pos
            act[:-1] *= 2

            next_obs, _, _, _ = env.step(act)

            if video_fn is not None:
                video_img = env.env.sim.render(height=512, width=512, camera_name="agentview")[::-1]
                video_writer.append_data(video_img)


        time_taken += time.time() - start


        normal_reward += env.get_reward()

    if video_fn is not None:
        video_writer.close()
    print(f"Joint position reward: {normal_reward}")
    print(f"Time take joint position: {time_taken}")
    print(f"Num joint position actions: {num_actions}")

    print()


if __name__ == "__main__":

    ### execute functions
    print("Replaying by aggregating without gripper check")
    replay_by_aggregating(demo_file, 100, video_fn="/media/nadun/Data/phd_project/robomimic/videos/lift_sped_up/aggregated_actions_100.mp4")
    print("Replaying by aggregating with gripper check")
    replay_by_aggregating(demo_file, 100, aggregating_function=aggregate_delta_actions_with_gripper_check,
                          video_fn="/media/nadun/Data/phd_project/robomimic/videos/lift_sped_up/aggregated_actions_with_gripper_check_100.mp4")
    # replay_by_skipping(demo_file, 100, video_fn="/media/nadun/Data/phd_project/robomimic/videos/lift_sped_up/skipping_actions_3_100.mp4")
    # replay_normal_speed(demo_file, 10, video_fn="/media/nadun/Data/phd_project/robomimic/videos/lift_sped_up/normal_10_absolute_actions_method_1.mp4")
    replay_joint_position_actions(demo_fn, 200, video_fn="/media/nadun/Data/phd_project/robomimic/videos/lift_sped_up/joint_positions_actions_200.mp4")






