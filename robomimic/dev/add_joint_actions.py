import nexusformat.nexus as nx
import numpy as np
import h5py
import robomimic.utils.transform_utils as T



demo_fn  = "/nethome/nkra3/flash7/phd_project/robomimic-nadun/datasets/tool_hang/ph/all_obs_v141.hdf5"
# demo = nx.nxload(demo_fn)
# print(demo.tree)

demo_f = h5py.File(demo_fn, "a")
demo_file = demo_f['data']
env_args = demo_file.attrs["env_args"]
print(env_args)
counter = 0

for demo in demo_file:
    if (counter % 10) == 0:
        print(f"Processing demo: {counter}")
    counter += 1
    demo = demo_file[demo]
    delta_actions = demo['actions'][:]
    joint_actions = demo['obs/robot0_joint_pos'][:]

    # Create joint position action
    

    # Create new joint action array with the gripper action included:
    gripper_act = delta_actions[:, -1, np.newaxis]
    joint_actions = np.concatenate((joint_actions, gripper_act), axis=1)

    demo.create_dataset("joint_actions", data=joint_actions)



demo_f.close()




