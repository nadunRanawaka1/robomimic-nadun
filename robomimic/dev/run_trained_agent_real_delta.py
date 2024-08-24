"""
The main script for evaluating a policy in an environment.

Args:
    agent (str): path to saved checkpoint pth file

    horizon (int): if provided, override maximum horizon of rollout from the one 
        in the checkpoint

    env (str): if provided, override name of env from the one in the checkpoint,
        and use it for rollouts

    render (bool): if flag is provided, use on-screen rendering during rollouts

    video_path (str): if provided, render trajectories to this video file path

    video_skip (int): render frames to a video every @video_skip steps

    camera_names (str or [str]): camera name(s) to use for rendering on-screen or to video

    dataset_path (str): if provided, an hdf5 file will be written at this path with the
        rollout data

    dataset_obs (bool): if flag is provided, and @dataset_path is provided, include 
        possible high-dimensional observations in output dataset hdf5 file (by default,
        observations are excluded and only simulator states are saved).

    seed (int): if provided, set seed for rollouts

Example usage:

    # Evaluate a policy with 50 rollouts of maximum horizon 400 and save the rollouts to a video.
    # Visualize the agentview and wrist cameras during the rollout.
    
    python run_trained_agent.py --agent /path/to/model.pth \
        --n_rollouts 50 --horizon 400 --seed 0 \
        --video_path /path/to/output.mp4 \
        --camera_names agentview robot0_eye_in_hand 

    # Write the 50 agent rollouts to a new dataset hdf5.

    python run_trained_agent.py --agent /path/to/model.pth \
        --n_rollouts 50 --horizon 400 --seed 0 \
        --dataset_path /path/to/output.hdf5 --dataset_obs 

    # Write the 50 agent rollouts to a new dataset hdf5, but exclude the dataset observations
    # since they might be high-dimensional (they can be extracted again using the
    # dataset_states_to_obs.py script).

    python run_trained_agent.py --agent /path/to/model.pth \
        --n_rollouts 50 --horizon 400 --seed 0 \
        --dataset_path /path/to/output.hdf5
"""
import argparse
import json
import pickle
import time

import h5py
import imageio
import numpy as np
from copy import deepcopy
import signal
import sys

import torch

import robomimic
import robomimic.utils.file_utils as FileUtils
import robomimic.utils.env_utils as EnvUtils
import robomimic.utils.torch_utils as TorchUtils
import robomimic.utils.tensor_utils as TensorUtils
import robomimic.utils.obs_utils as ObsUtils
from robomimic.envs.env_base import EnvBase
from robomimic.envs.wrappers import EnvWrapper
from robomimic.algo import RolloutPolicy
from robomimic.dev.dev_utils import demo_obs_to_obs_dict

import matplotlib.pyplot as plt

import h5py
# import nexusformat.nexus as nx

# demo_fn = "/mnt/Data/atrp_ac_learning/demos/merged_demos/demo_foam_cube_generalizable.hdf5"
# # file = nx.nxload(demo_fn)
# demo = h5py.File(demo_fn)
# demo = demo['data']['demo_1']
# actions = demo['delta_actions']
# rec_obs = demo['obs']



def rollout(
        policy, env, horizon, render=False, video_writer=None, video_skip=5,
        return_obs=False, camera_names=None, real=False,
        rollout_demo=False, rollout_demo_obs=False, demo=None, demo_act_key="absolute_axis_angle_actions"):
    """
    Helper function to carry out rollouts. Supports on-screen rendering, off-screen rendering to a video, 
    and returns the rollout trajectory.

    Args:
        policy (instance of RolloutPolicy): policy loaded from a checkpoint
        env (instance of EnvBase): env loaded from a checkpoint or demonstration metadata
        horizon (int): maximum horizon for the rollout
        render (bool): whether to render rollout on-screen
        video_writer (imageio writer): if provided, use to write rollout to video
        video_skip (int): how often to write video frames
        return_obs (bool): if True, return possibly high-dimensional observations along the trajectoryu. 
            They are excluded by default because the low-dimensional simulation states should be a minimal 
            representation of the environment. 
        camera_names (list): determines which camera(s) are used for rendering. Pass more than
            one to output a video with multiple camera views concatenated horizontally.
        real (bool): if real robot rollout
        rollout_demo: to use actions from demo instead of actual actions
        rollout_demo_obs: to use obs from the demo and rollout instead of actual obs
        demo: demo to use for rolling out actions or obs
        demo_act_key: which actions to rollout from demo if rolling out demo actions

    Returns:
        stats (dict): some statistics for the rollout - such as return, horizon, and task success
        traj (dict): dictionary that corresponds to the rollout trajectory
    """
    assert isinstance(env, EnvBase)  or isinstance(env, EnvWrapper)
    assert isinstance(policy, RolloutPolicy)
    assert not (render and (video_writer is not None))
    if rollout_demo or rollout_demo_obs:
        assert demo is not None , "Please provide demo_fn to rollout demo"

    if rollout_demo:
        print("ROLLING OUT DEMO")
    if rollout_demo_obs:
        print("ROLLING OUT MODEL WITH DEMO OBS")

    if rollout_demo or rollout_demo_obs: # If rolling out demo, only step for as many actions are in the demo
        horizon = demo[demo_act_key][:].shape[0]

    policy.start_episode()
    obs = env.reset()

    if real:
        input("ready for next eval? hit enter to continue")
    state_dict = dict()

    if not real:
        state_dict = env.get_state()

        # hack that is necessary for robosuite tasks for deterministic action playback
        obs = env.reset_to(state_dict)


    video_count = 0  # video frame counter
    total_reward = 0.
    traj = dict(actions=[], rewards=[], dones=[], states=[], initial_state_dict=state_dict)
    diff = np.array([0, 0, 0, 0, 0, 0, 0, 0], dtype=float)

    # TODO: MOVE THIS
    kwargs = {"return_action_sequence": False}
    if return_obs:
        # store observations too
        traj.update(dict(obs=[], next_obs=[]))
    try:
        for step_i in range(horizon):
            start = time.time()
            if rollout_demo_obs:
                obs = demo_obs_to_obs_dict(demo['obs'], step_i)
            else:
                obs = env.get_observation()
            # get action from policy
            act = policy(ob=obs, **kw_args)
            np.set_printoptions(precision=8)
            print("Run_Trained_Agent_real: action from policy is: {}".format(act))

            if rollout_demo:
                demo_act = demo[demo_act_key][step_i]
                print("action from demo: {}".format(demo_act))
                diff += np.abs(act - demo_act)
                print("diff is {}".format(diff))
                act = demo_act

            # play action
            next_obs, r, done, _ = env.step(act, **kwargs)

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
                        if real:
                            video_img.append(env.render(mode="rgb_array", camera_name=cam_name))
                        else:
                            video_img.append(env.render(mode="rgb_array", height=512, width=512, camera_name=cam_name))
                    video_img = np.concatenate(video_img, axis=1) # concatenate horizontally
                    video_writer.append_data(video_img)
                video_count += 1

            # collect transition
            traj["actions"].append(act)
            traj["rewards"].append(r)
            traj["dones"].append(done)
            if not real:
                traj["states"].append(state_dict["states"])
            if return_obs:
                # Note: We need to "unprocess" the observations to prepare to write them to dataset.
                #       This includes operations like channel swapping and float to uint8 conversion
                #       for saving disk space.
                traj["obs"].append(ObsUtils.unprocess_obs(obs))
                traj["next_obs"].append(ObsUtils.unprocess_obs(next_obs))
            end = time.time()
            # time.sleep(0.02)
            print (f"Robot control frequency : {1/(end-start)}")
            # break if done or if success
            if done or success:
                break

            # update for next iter
            obs = deepcopy(next_obs)
            if not real:
                state_dict = env.get_state()


    except env.rollout_exceptions as e:
        print("WARNING: got rollout exception {}".format(e))

    stats = dict(Return=total_reward, Horizon=(step_i + 1), Success_Rate=float(success))

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

def rollout_with_action_sequence(policy, env, horizon, render=False, video_writer=None, video_skip=5,
        return_obs=False, camera_names=None, real=False,
        rollout_demo=False, rollout_demo_obs=False, demo=None, demo_act_key="delta_joint_positions"):
    """
    Helper function to carry out rollouts. Supports on-screen rendering, off-screen rendering to a video,
    and returns the rollout trajectory.

    Args:
        policy (instance of RolloutPolicy): policy loaded from a checkpoint
        env (instance of EnvBase): env loaded from a checkpoint or demonstration metadata
        horizon (int): maximum horizon for the rollout
        render (bool): whether to render rollout on-screen
        video_writer (imageio writer): if provided, use to write rollout to video
        video_skip (int): how often to write video frames
        return_obs (bool): if True, return possibly high-dimensional observations along the trajectoryu.
            They are excluded by default because the low-dimensional simulation states should be a minimal
            representation of the environment.
        camera_names (list): determines which camera(s) are used for rendering. Pass more than
            one to output a video with multiple camera views concatenated horizontally.
        real (bool): if real robot rollout
        rollout_demo: to use actions from demo instead of actual actions
        rollout_demo_obs: to use obs from the demo and rollout instead of actual obs
        demo: demo to use for rolling out actions or obs
        demo_act_key: which actions to rollout from demo if rolling out demo actions

    Returns:
        stats (dict): some statistics for the rollout - such as return, horizon, and task success
        traj (dict): dictionary that corresponds to the rollout trajectory
    """
    assert isinstance(env, EnvBase)  or isinstance(env, EnvWrapper)
    assert isinstance(policy, RolloutPolicy)
    assert not (render and (video_writer is not None))
    if rollout_demo or rollout_demo_obs:
        assert demo is not None , "Please provide demo_fn to rollout demo"
        horizon = demo[demo_act_key][:].shape[0] # If rolling out demo, only step for as many actions are in the demo


    if rollout_demo:
        print("ROLLING OUT DEMO")
    if rollout_demo_obs:
        print("ROLLING OUT MODEL WITH DEMO OBS")


    policy.start_episode()
    obs = env.reset()

    if real:
        input("ready for next eval? hit enter to continue")
        time.sleep(3)
    state_dict = dict()

    if not real:
        state_dict = env.get_state()

        # hack that is necessary for robosuite tasks for deterministic action playback
        obs = env.reset_to(state_dict)


    video_count = 0  # video frame counter
    total_reward = 0.
    traj = dict(actions=[], rewards=[], dones=[], states=[], initial_state_dict=state_dict)
    diff = np.array([0, 0, 0, 0, 0, 0, 0], dtype=float)
    step_i = 0

    # TODO: MOVE THIS
    # kwargs = {"return_action_sequence": True, "step_action_sequence": False,
    #           "control_mode": "Joint_Position"}
    kwargs = {"return_action_sequence": True, "step_action_sequence": True,
              "control_mode": "Joint_Position_Trajectory"}

    env.robot_interface.switch_to_joint_traj_controller()

    action_sequence_length = policy.policy.algo_config.horizon.action_horizon
    run_actions = 20

    env.robot_interface.reset_joint_position_messages()
    inf_time_list = []
    traj_publish_times = []
    inf_durations = []
    if return_obs:
        # store observations too
        traj.update(dict(obs=[], next_obs=[]))
    try:
        while step_i < horizon:
            start = time.time()
            if rollout_demo_obs:
                obs = demo_obs_to_obs_dict(demo['obs'], step_i)
                obs = env.get_observation(obs)
            else:
                obs = env.get_observation()
            # get action from policy
            kwargs["inf_start_time"] = env.robot_interface.node.get_clock().now().to_msg()
            start_inf = time.time()
            act = policy(ob=obs, **kwargs) # Act will be sequence (N, act_dim)

            # TODO SINCE THIS IS DELTA JOINT POSITION, WE ADD THE CURRENT JOINT POSITION TO THE MODEL PREDS
            new_act = np.cumsum(act, axis=0)
            new_act[:, -1] = act[:, -1]
            new_act[:, :-1] += obs['joint_positions']
            act = np.copy(new_act)

            if step_i == 0: # TODO Find a better way of handling this
                kwargs["inf_start_time"] = env.robot_interface.node.get_clock().now().to_msg()

            inf_dur = time.time() - start
            # print(f"Time taken for inf: {inf_dur}")
            inf_durations.append(inf_dur)

            np.set_printoptions(precision=8)

            if rollout_demo:
                demo_act = demo[demo_act_key][step_i: step_i+ action_sequence_length]
                # diff += np.abs(act - demo_act)
                # print("diff is {}".format(diff))
                act = demo_act
                kwargs["inf_start_time"] = env.robot_interface.node.get_clock().now().to_msg()

            if rollout_demo_obs:
                demo_act = demo[demo_act_key][step_i: step_i + action_sequence_length]
                # diff += np.abs(act - demo_act)
                # print("diff is {}".format(diff))
                kwargs["inf_start_time"] = env.robot_interface.node.get_clock().now().to_msg()

            inf_time_list.append(kwargs["inf_start_time"])


            if not kwargs['step_action_sequence']:
                for i in range(act.shape[0]):
                    a = act[i]
                    start = time.time()
                    kwargs["start_time"] = start
                    next_obs, r, done, _ = env.step(a, need_obs=False, **kwargs)
                    time.sleep(env.robot_interface.j_point_time)
            else:
                # play action
                next_obs, r, done, _ = env.step(act, **kwargs)
                if rollout_demo or rollout_demo_obs:
                    time.sleep(action_sequence_length*env.robot_interface.j_point_time)
                else:
                    # time.sleep((action_sequence_length)*env.robot_interface.j_point_time)
                    time.sleep(run_actions * env.robot_interface.j_point_time)

            # compute reward
            total_reward += r
            success = env.is_success()["task"]

            traj_publish_times.append(env.robot_interface.get_previous_traj_publish_time())

            # visualization
            if render:
                env.render(mode="human", camera_name=camera_names[0])
            if video_writer is not None:
                if video_count % video_skip == 0:
                    video_img = []
                    for cam_name in camera_names:
                        if real:
                            video_img.append(env.render(mode="rgb_array", camera_name=cam_name))
                        else:
                            video_img.append(env.render(mode="rgb_array", height=512, width=512, camera_name=cam_name))
                    video_img = np.concatenate(video_img, axis=1) # concatenate horizontally
                    video_writer.append_data(video_img)
                video_count += 1

            # collect transition
            traj["actions"].append(act)
            traj["rewards"].append(r)
            traj["dones"].append(done)
            if not real:
                traj["states"].append(state_dict["states"])
            elif real:
                traj["states"].append(obs["joint_positions"])
            if return_obs:
                # Note: We need to "unprocess" the observations to prepare to write them to dataset.
                #       This includes operations like channel swapping and float to uint8 conversion
                #       for saving disk space.
                traj["obs"].append(ObsUtils.unprocess_obs(obs))
                traj["next_obs"].append(ObsUtils.unprocess_obs(next_obs))
            end = time.time()
            # time.sleep(0.02)
            print (f"Robot control frequency : {1/(end-start)}")
            # break if done or if success
            if done or success:
                break

            # update for next iter
            obs = deepcopy(next_obs)
            if not real:
                state_dict = env.get_state()

            # Update step_i
            # step_i += action_sequence_length
            step_i += run_actions

    except env.rollout_exceptions as e:
        print("WARNING: got rollout exception {}".format(e))

    joint_position_messages = env.robot_interface.get_joint_positions_messages()

    stats = dict(Return=total_reward, Horizon=(step_i + 1), Success_Rate=float(success))

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

    traj["inf_start_times"] = inf_time_list
    traj["traj_publish_times"] = traj_publish_times
    traj["all_joint_msg"] = joint_position_messages
    traj['traj_point_time'] = env.robot_interface.j_point_time
    traj['run_actions'] = run_actions
    traj["inf_durations"] = inf_durations
    return stats, traj

def run_trained_agent(args):

    # some arg checking
    write_video = (args.video_path is not None)
    assert not (args.render and write_video) # either on-screen or video but not both
    if args.render:
        # on-screen rendering can only support one camera
        assert len(args.camera_names) == 1

    # relative path to agent
    ckpt_path = args.agent

    # device
    device = TorchUtils.get_torch_device(try_to_use_cuda=True)

    # restore policy
    _, ckpt_dict = FileUtils.policy_from_checkpoint(ckpt_path=ckpt_path, device=device, verbose=True)
    policy, _ = FileUtils.policy_from_checkpoint(ckpt_path=ckpt_path, device=device, verbose=True)

    # read rollout settings
    rollout_num_episodes = args.n_rollouts
    rollout_horizon = args.horizon
    if rollout_horizon is None:
        # read horizon from config
        config, _ = FileUtils.config_from_checkpoint(ckpt_dict=ckpt_dict)
        rollout_horizon = config.experiment.rollout.horizon

    # create environment from saved checkpoint

        #TODO adding in hardcoded metadata for UR5 env, implement this correctly in the data collection loop
        # ckpt_dict['env_metadata']['type'] = 5
        # ckpt_dict['env_metadata']["env_kwargs"] = {}
        # ckpt_dict['env_metadata']['env_name'] = "UR5_REAL_TYPE"
        env, _ = FileUtils.env_from_checkpoint(
        ckpt_dict=ckpt_dict, 
        env_name=args.env, 
        render=args.render, 
        render_offscreen=(args.video_path is not None), 
        verbose=True,
    )

    is_real_robot = EnvUtils.is_real_robot_env(env=env) or EnvUtils.is_real_robot_gprs_env(env=env)

    #Set obs shapes for policy: #TODO delete this later
    env.set_obs_shapes(policy.policy.obs_shapes)

    # maybe set seed
    if args.seed is not None:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)

    # maybe create video writer
    video_writer = None
    if write_video:
        video_writer = imageio.get_writer(args.video_path, fps=20)

    # maybe open hdf5 to write rollouts
    write_dataset = (args.dataset_path is not None)
    if write_dataset:
        # data_writer = h5py.File(args.dataset_path, "w")
        # data_grp = data_writer.create_group("data")
        # total_samples = 0
        rollout_data = {}

    rollout_stats = []

    rollout_horizon = 200 #TODO remove this


    for i in range(rollout_num_episodes):
        try:
            # TODO ### ROLLING OUT NORMALLY
            # stats, traj = rollout(
            #     policy=policy,
            #     env=env,
            #     horizon=rollout_horizon,
            #     render=args.render,
            #     video_writer=video_writer,
            #     video_skip=args.video_skip,
            #     return_obs=(write_dataset and args.dataset_obs),
            #     camera_names=args.camera_names,
            #     real=is_real_robot,
            # )

            # TODO ROLLING OUT AS A SEQUENCE
            stats, traj = rollout_with_action_sequence(
                policy=policy,
                env=env,
                horizon=rollout_horizon,
                render=args.render,
                video_writer=video_writer,
                video_skip=args.video_skip,
                return_obs=(write_dataset and args.dataset_obs),
                camera_names=args.camera_names,
                real=is_real_robot,
            )
        except KeyboardInterrupt:
            if is_real_robot:
                print("ctrl-C catched, stop execution")
                print("env rate measure")
                print(env.rate_measure)
                continue
            else:
                sys.exit(0)

        rollout_stats.append(stats)

        if write_dataset:
            # store transitions

            rollout_data["demo_{}".format(i)] = traj

            # rollout_data["demo_{}".format(i)]["actions"] = traj["actions"]
            # rollout_data["demo_{}".format(i)]["states"] = traj["states"]
            # rollout_data["demo_{}".format(i)]["rewards"] = traj["rewards"]
            # rollout_data["demo_{}".format(i)]["dones"] = traj["dones"]
            #
            # rollout_data["demo_{}".format(i)]["traj_start_times"] = traj["traj_start_times"]

            # ep_data_grp = data_grp.create_group("demo_{}".format(i))
            # ep_data_grp.create_dataset("actions", data=np.array(traj["actions"]))
            # ep_data_grp.create_dataset("states", data=np.array(traj["states"]))
            # ep_data_grp.create_dataset("rewards", data=np.array(traj["rewards"]))
            # ep_data_grp.create_dataset("dones", data=np.array(traj["dones"]))
            # if args.dataset_obs:
            #     rollout_data["demo_{}".format(i)]["obs"] = {}
            #     for k in traj["obs"]:
            #         rollout_data["demo_{}".format(i)]["obs"][k] = traj["obs"][k]

                    # ep_data_grp.create_dataset("obs/{}".format(k), data=np.array(traj["obs"][k]))
                    # ep_data_grp.create_dataset("next_obs/{}".format(k), data=np.array(traj["next_obs"][k]))

            # rollout_data["demo_{}".format(i)]["obs"]["all_joint_msg"] = traj["all_joint_msg"]
            # episode metadata
            # if "model" in traj["initial_state_dict"]:
            #     ep_data_grp.attrs["model_file"] = traj["initial_state_dict"]["model"] # model xml for this episode
            # ep_data_grp.attrs["num_samples"] = traj["actions"].shape[0] # number of transitions in this episode
            # total_samples += traj["actions"].shape[0]

    rollout_stats = TensorUtils.list_of_flat_dict_to_dict_of_list(rollout_stats)
    avg_rollout_stats = { k : np.mean(rollout_stats[k]) for k in rollout_stats }
    avg_rollout_stats["Num_Success"] = np.sum(rollout_stats["Success_Rate"])
    print("Average Rollout Stats")
    print(json.dumps(avg_rollout_stats, indent=4))

    if write_video:
        video_writer.close()

    if write_dataset:
        # global metadata
        # data_grp.attrs["total"] = total_samples
        # data_grp.attrs["env_args"] = json.dumps(env.serialize(), indent=4) # environment info
        # data_writer.close()
        with open(args.dataset_path, "wb") as f:
            pickle.dump(rollout_data, f)
        print("Wrote dataset trajectories to {}".format(args.dataset_path))


def run_trained_agent_with_demo(args):

    # TODO: SETTING UP DEMO STUFF, PASS THIS IN TO THE FUNCTION:
    demo_fn = "/home/robot-aiml/ac_learning_repos/Task_Demos/merged/demo_put_strawberry_in_bowl.hdf5"

    demo_file = h5py.File(demo_fn)
    demos = demo_file['data']


    # some arg checking
    write_video = (args.video_path is not None)
    assert not (args.render and write_video) # either on-screen or video but not both
    if args.render:
        # on-screen rendering can only support one camera
        assert len(args.camera_names) == 1

    # relative path to agent
    ckpt_path = args.agent

    # device
    device = TorchUtils.get_torch_device(try_to_use_cuda=True)

    # restore policy
    _, ckpt_dict = FileUtils.policy_from_checkpoint(ckpt_path=ckpt_path, device=device, verbose=True)
    policy, _ = FileUtils.policy_from_checkpoint(ckpt_path=ckpt_path, device=device, verbose=True)

    # read rollout settings
    rollout_num_episodes = args.n_rollouts
    rollout_horizon = args.horizon
    if rollout_horizon is None:
        # read horizon from config
        config, _ = FileUtils.config_from_checkpoint(ckpt_dict=ckpt_dict)
        rollout_horizon = config.experiment.rollout.horizon

    # create environment from saved checkpoint
        env, _ = FileUtils.env_from_checkpoint(
        ckpt_dict=ckpt_dict,
        env_name=args.env,
        render=args.render,
        render_offscreen=(args.video_path is not None),
        verbose=True,
    )

    is_real_robot = EnvUtils.is_real_robot_env(env=env) or EnvUtils.is_real_robot_gprs_env(env=env)

    #Set obs shapes for policy: #TODO delete this later
    env.set_obs_shapes(policy.policy.obs_shapes)

    # maybe set seed
    if args.seed is not None:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)

    # maybe create video writer
    video_writer = None
    if write_video:
        video_writer = imageio.get_writer(args.video_path, fps=20)

    # maybe open hdf5 to write rollouts
    write_dataset = (args.dataset_path is not None)
    if write_dataset:
        data_writer = h5py.File(args.dataset_path, "w")
        data_grp = data_writer.create_group("data")
        total_samples = 0

    rollout_stats = []

    rollout_horizon = 300 #TODO remove this

    i = 0

    for demo in demos:

        try:
            # TODO ### ROLLING OUT NORMALLY
            # stats, traj = rollout(
            #     policy=policy,
            #     env=env,
            #     horizon=rollout_horizon,
            #     render=args.render,
            #     video_writer=video_writer,
            #     video_skip=args.video_skip,
            #     return_obs=(write_dataset and args.dataset_obs),
            #     camera_names=args.camera_names,
            #     real=is_real_robot,
            # )

            # TODO ROLLING OUT AS A SEQUENCE
            demo = demos[demo]
            stats, traj = rollout_with_action_sequence(
                policy=policy,
                env=env,
                horizon=rollout_horizon,
                render=args.render,
                video_writer=video_writer,
                video_skip=args.video_skip,
                return_obs=(write_dataset and args.dataset_obs),
                camera_names=args.camera_names,
                real=is_real_robot,
                rollout_demo_obs=True,
                demo=demo,
                demo_act_key="joint_position_actions"

            )
        except KeyboardInterrupt:
            if is_real_robot:
                print("ctrl-C catched, stop execution")
                print("env rate measure")
                print(env.rate_measure)
                continue
            else:
                sys.exit(0)

        rollout_stats.append(stats)

        if write_dataset:
            # store transitions
            ep_data_grp = data_grp.create_group("demo_{}".format(i))
            ep_data_grp.create_dataset("actions", data=np.array(traj["actions"]))
            ep_data_grp.create_dataset("states", data=np.array(traj["states"]))
            ep_data_grp.create_dataset("rewards", data=np.array(traj["rewards"]))
            ep_data_grp.create_dataset("dones", data=np.array(traj["dones"]))
            if args.dataset_obs:
                for k in traj["obs"]:
                    ep_data_grp.create_dataset("obs/{}".format(k), data=np.array(traj["obs"][k]))
                    ep_data_grp.create_dataset("next_obs/{}".format(k), data=np.array(traj["next_obs"][k]))

            # episode metadata
            if "model" in traj["initial_state_dict"]:
                ep_data_grp.attrs["model_file"] = traj["initial_state_dict"]["model"] # model xml for this episode
            ep_data_grp.attrs["num_samples"] = traj["actions"].shape[0] # number of transitions in this episode
            total_samples += traj["actions"].shape[0]

        i += 1



    rollout_stats = TensorUtils.list_of_flat_dict_to_dict_of_list(rollout_stats)
    avg_rollout_stats = { k : np.mean(rollout_stats[k]) for k in rollout_stats }
    avg_rollout_stats["Num_Success"] = np.sum(rollout_stats["Success_Rate"])
    print("Average Rollout Stats")
    print(json.dumps(avg_rollout_stats, indent=4))

    if write_video:
        video_writer.close()

    if write_dataset:
        # global metadata
        data_grp.attrs["total"] = total_samples
        data_grp.attrs["env_args"] = json.dumps(env.serialize(), indent=4) # environment info
        data_writer.close()
        print("Wrote dataset trajectories to {}".format(args.dataset_path))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Path to trained model
    parser.add_argument(
        "--agent",
        type=str,
        # required=True,
        help="path to saved checkpoint pth file",
    )

    # number of rollouts
    parser.add_argument(
        "--n_rollouts",
        type=int,
        default=2,
        help="number of rollouts",
    )

    # maximum horizon of rollout, to override the one stored in the model checkpoint
    parser.add_argument(
        "--horizon",
        type=int,
        default=None,
        help="(optional) override maximum horizon of rollout from the one in the checkpoint",
    )

    # Env Name (to override the one stored in model checkpoint)
    parser.add_argument(
        "--env",
        type=str,
        default=None,
        help="(optional) override name of env from the one in the checkpoint, and use\
            it for rollouts",
    )

    # Whether to render rollouts to screen
    parser.add_argument(
        "--render",
        action='store_true',
        help="on-screen rendering",
    )

    # Dump a video of the rollouts to the specified path
    parser.add_argument(
        "--video_path",
        type=str,
        default=None,
        help="(optional) render rollouts to this video file path",
    )

    # How often to write video frames during the rollout
    parser.add_argument(
        "--video_skip",
        type=int,
        default=1,
        help="render frames to video every n steps",
    )

    # camera names to render
    parser.add_argument(
        "--camera_names",
        type=str,
        nargs='+',
        default=["sideview_left_camera_rgb"],
        help="(optional) camera name(s) to use for rendering on-screen or to video",
    )

    # If provided, an hdf5 file will be written with the rollout data
    parser.add_argument(
        "--dataset_path",
        type=str,
        default=None,
        help="(optional) if provided, an hdf5 file will be written at this path with the rollout data",
    )

    # If True and @dataset_path is supplied, will write possibly high-dimensional observations to dataset.
    parser.add_argument(
        "--dataset_obs",
        action='store_true',
        help="include possibly high-dimensional observations in output dataset hdf5 file (by default,\
            observations are excluded and only simulator states are saved)",
    )

    # for seeding before starting rollouts
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="(optional) set seed for rollouts",
    )

    args = parser.parse_args()

    # args.agent= "/home/robot-aiml/ac_learning_repos/robomimic-nadun/bc_trained_models/real_robot/strawberry/strawberry_absolute_axis_actions_image_only/20240808154229/models/model_epoch_600.pth"
    args.agent= "/home/robot-aiml/ac_learning_repos/robomimic-nadun/bc_trained_models/real_robot/strawberry/strawberry_delta_joint_actions/20240822192410/models/model_epoch_250.pth"

    args.dataset_path = "/home/robot-aiml/ac_learning_repos/robomimic-nadun/bc_trained_models/real_robot/strawberry/strawberry_delta_joint_actions/20240822192410/logs/rollout_4x.pkl"

    run_trained_agent(args)
    #
    # run_trained_agent_with_demo(args)
