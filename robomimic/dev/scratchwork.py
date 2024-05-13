import os
import torch
import nexusformat.nexus as nx

demo_fn = "/media/nadun/Data/phd_project/robomimic/datasets/lift/ph/all_obs_v141.hdf5"
demo_file = nx.nxload(demo_fn)
print(demo_file.tree)

print()