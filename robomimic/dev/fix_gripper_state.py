import h5py
import numpy as np
import nexusformat.nexus as nx

demo_fn = "/home/robot-aiml/ac_learning_repos/Task_Demos/merged/move_to_skynet/demo_pick_cube_realsense.hdf5"
demo_file = h5py.File(demo_fn, 'a')

dataset_grp = demo_file['data']

for demo in dataset_grp:
    print(f"processing demo: {demo}")
    demo = dataset_grp[demo]
    gripper_state = demo['obs/gripper_state'][:]
    gripper_pos = gripper_state[..., np.newaxis]
    if "gripper_pos" in demo['obs']:
        del demo['obs/gripper_pos']
    demo.create_dataset('obs/gripper_pos', data=np.array(gripper_pos))

demo_file.close()

demo_file = nx.nxload(demo_fn)
print(demo_file.tree)