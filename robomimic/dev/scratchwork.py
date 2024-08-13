import os
import nexusformat.nexus as nx
import h5py

demo_fn = "/media/nadun/Data/phd_project/robomimic/datasets/can/ph/all_obs_v141_with_aggregated_actions.hdf5"
demo_file = nx.nxload(demo_fn)
print(demo_file.tree)

demo_file = h5py.File(demo_fn)
data = demo_file['data']
demo = data['demo_5']
act1 = demo['agg_actions_magnitude_1'][:]
act2 = demo['agg_actions_magnitude_2'][:]
act3 = demo['agg_actions_magnitude_3'][:]
act4 = demo['agg_actions_magnitude_4'][:]


print()