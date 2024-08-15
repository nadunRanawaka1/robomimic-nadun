import os
import nexusformat.nexus as nx
import h5py

demo_fn = "/media/nadun/Data/phd_project/robomimic/datasets/can/ph/all_obs_v141_with_aggregated_actions.hdf5"
demo_file = nx.nxload(demo_fn)
print(demo_file.tree)

print()