import os
import nexusformat.nexus as nx

demo_fn = "/nethome/nkra3/flash7/phd_project/robomimic-nadun/datasets/square/ph/all_obs_v141.hdf5"
demo_file = nx.nxload(demo_fn)
print(demo_file.tree)

print()