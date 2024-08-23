import os
import nexusformat.nexus as nx
import h5py

demo_fn = "/home/robot-aiml/ac_learning_repos/robomimic-nadun/bc_trained_models/real_robot/strawberry/strawberry_joint_position_actions/20240808154101/logs/rollout_joint_position_trajectory_large_delay_8_steps.hdf5"
demo_file = nx.nxload(demo_fn)
print(demo_file.tree)

print()