import nexusformat.nexus as nx
import numpy as np
import h5py
import robomimic.utils.transform_utils as T
import robomimic.utils.file_utils as FileUtils
from dev_utils import complete_setup_for_replay, create_absolute_actions
from copy import deepcopy
import argparse


def delta_axis_angle_to_absolute(delta_angles, quats):
    ret = []
    for i in range(quats.shape[0]):
        quat = quats[i]
        delta_aa = delta_angles[i]

        start_rot = T.quat2mat(quat)
        axis, angle = T.vec2axisangle(delta_aa)
        delta_rot = T.axisangle2mat(axis, angle)
        abs_rot_mat = np.dot(delta_rot, start_rot)  # Final absolute rot mat

        abs_axis, abs_angle = T.mat2axisangle(abs_rot_mat)
        abs_aa = T.axisangle2vec(abs_axis, abs_angle)
        if any(np.isnan(abs_aa)):
            print()
        ret.append(abs_aa)

    return np.array(ret)

def add_actions_to_dataset(demo_fn):

    print(f"Processing dataset: {demo_fn}")

    demo_f = h5py.File(demo_fn, "a")
    demo_file = demo_f['data']

    # Create env for absolute actions generation
    env, _ = complete_setup_for_replay(demo_fn)

    counter = 0

    for demo in demo_file:
        if (counter % 10) == 0:
            print(f"Processing demo: {counter}")
        counter += 1
        demo = demo_file[demo]

        ### Cleanup everything first
        if "absolute_actions" in demo:
            del demo["absolute_actions"]
        if "joint_actions" in demo:
            del demo["joint_actions"]
        if "joint_position_actions" in demo:
            del demo["joint_position_actions"]
        if "delta_joint_actions" in demo:
            del demo["delta_joint_actions"]

        delta_actions = demo['actions'][:]
        states = demo['states'][:]

        ### Create absolute eef actions

        absolute_actions = create_absolute_actions(states, actions=delta_actions, env=env)

        # # Create absolute position action
        # delta_pos = delta_actions[:, 0:3]
        # pos = demo['obs/robot0_eef_pos'][:]
        # target_pos = delta_pos + pos
        #
        # # Create absolute axis angle rotation action
        # delta_aa = delta_actions[:, 3:6]
        # quat = demo['obs/robot0_eef_quat'][:]
        # target_axis_angles = delta_axis_angle_to_absolute(delta_aa, quat)
        #
        # # Create new absolute action array:
        # gripper_act = delta_actions[:, -1, np.newaxis]
        # absolute_actions = np.concatenate((target_pos, target_axis_angles, gripper_act), axis=1)

        demo.create_dataset("absolute_actions", data=absolute_actions)

        ### Create joint position action
        joint_actions = demo['obs/robot0_joint_pos'][1:] # We offset by 1 since the next joint position is the action
        last_joint_action = np.copy(demo['obs/robot0_joint_pos'][-1:, :])
        joint_actions = np.concatenate([joint_actions, last_joint_action], axis=0)

        # Create new joint action array with the gripper action included:
        gripper_act = delta_actions[:, -1, np.newaxis]
        joint_actions = np.concatenate((joint_actions, gripper_act), axis=1)
        demo.create_dataset("joint_position_actions", data=joint_actions)

        # Creating delta joint actions
        delta_joint_actions = deepcopy(demo['joint_position_actions'][:])

        joint_obs = demo['obs/robot0_joint_pos'][:]
        delta_joint_actions[:, :-1] = delta_joint_actions[:, :-1] - joint_obs
        demo.create_dataset("delta_joint_actions", data=delta_joint_actions)

    demo_f.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Path to trained model
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="path to dataset",
    )
    args = parser.parse_args()

    add_actions_to_dataset(args.dataset)




