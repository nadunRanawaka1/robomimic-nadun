import os
import nexusformat.nexus as nx
import h5py
from robomimic.robosuite_configs.paths import baxter
#
# demo_fn = "/home/robot-aiml/ac_learning_repos/Task_Demos/merged/demo_put_strawberry_in_bowl.hdf5"
# demo_file = nx.nxload(demo_fn)
# print(demo_file.tree)
# demo_file = h5py.File(demo_fn)
# gripper_state = demo_file['data/demo_5/obs/gripper_state'][:]

demo_fn = "/media/nadun/Data/phd_project/robomimic/datasets/lift/ph/demo_v141.hdf5"
demo_fn = "/nethome/nkra3/flash7/phd_project/robomimic-nadun/datasets/can/ph/all_obs_v141.hdf5"


demo_file = nx.nxload(demo_fn)
print(demo_file.tree)

