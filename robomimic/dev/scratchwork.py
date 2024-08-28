import os
import nexusformat.nexus as nx
import h5py

demo_fn = "/home/robot-aiml/ac_learning_repos/Task_Demos/merged/demo_put_strawberry_in_bowl.hdf5"
demo_file = nx.nxload(demo_fn)
print(demo_file.tree)
demo_file = h5py.File(demo_fn)
data = demo_file['data']
demo = data['demo_10']
obs = demo['obs']
gripper_state = obs['gripper_state'][:]
joint_obs = obs['joint_positions'][:]
joint_actions = demo['joint_position_actions'][:]

print()

