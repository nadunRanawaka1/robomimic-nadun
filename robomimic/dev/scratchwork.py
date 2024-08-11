import os
import nexusformat.nexus as nx
import h5py

demo_fn = "/media/nadun/Data/phd_project/robomimic/datasets/lift/ph/all_obs_v141_aggregated_actions_3.hdf5"
demo_file = nx.nxload(demo_fn)
print(demo_file.tree)

demo_file = h5py.File(demo_fn)
actions = demo_file['data/demo_50/actions'][:]

print()