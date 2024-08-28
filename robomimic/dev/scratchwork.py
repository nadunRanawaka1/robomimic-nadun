import os
import nexusformat.nexus as nx
import h5py

demo_fn = "/home/robot-aiml/ac_learning_repos/Task_Demos/merged/demo_put_strawberry_in_bowl.hdf5"
demo_file = nx.nxload(demo_fn)
print(demo_file.tree)
demo_file = h5py.File(demo_fn)
gripper_state = demo_file['data/demo_5/obs/gripper_state'][:]

print()

