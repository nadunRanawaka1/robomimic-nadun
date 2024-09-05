import h5py
import numpy as np
import nexusformat.nexus as nx

demo_fn = "/nethome/nkra3/flash7/ATRP_AC_Learning/datasets/speed_up_demos/demo_place_bolt_in_hole.hdf5"
demo_file = h5py.File(demo_fn, 'a')

dataset_grp = demo_file['data']

for demo in dataset_grp:
    print(f"processing demo: {demo}")
    demo = dataset_grp[demo]
    gripper_state = demo['obs/gripper_state'][:]
    gripper_pos = gripper_state[..., np.newaxis]
    demo.create_dataset('obs/gripper_pos', data=np.array(gripper_pos))

demo_file.close()

demo_file = nx.nxload(demo_fn)
print(demo_file.tree)