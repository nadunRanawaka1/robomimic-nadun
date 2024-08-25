from os import replace

import nexusformat.nexus as nx
import h5py
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
import numpy as np

demo_fn = "/media/nadun/Data/phd_project/robomimic/bc_trained_models/real_robot/logs/rollout_joint_position_trajectory_traj_replacement_2x_speed.hdf5"
# demo_fn = "/media/nadun/Data/phd_project/robomimic/bc_trained_models/real_robot/logs/rollout_joint_position_single.hdf5"
# demo_fn = "/media/nadun/Data/phd_project/robomimic/bc_trained_models/real_robot/logs/rollout_joint_position_trajectory_with_demo_obs.hdf5"

demo_file = nx.nxload(demo_fn)
print(demo_file.tree)

demo_file = h5py.File(demo_fn)
demo = demo_file['data/demo_0']
#
# preds = demo['actions'][:]
# timestep  = 0
# x_list = []
# y_list = []
# for i in range(preds.shape[0]):
#     traj =preds[i]
#     y = []
#     x = []
#     for j in range(traj.shape[0]):
#         y.append(traj[j, 0])
#         x.append(timestep)
#         timestep += 1
#     x_list.append(x)
#     y_list.append(y)
#
# color = iter(cm.rainbow(np.linspace(0, 1, len(y_list))))
# for i, traj in enumerate(y_list):
#     # c = next(color)
#     plt.plot(x_list[i], traj)
#
# plt.ylabel("Joint command (rads)")
# plt.xlabel("Timestep")
# plt.show()
# print()



#### PLOTTING WITH TRAJ REPLACEMENT

# preds = demo['actions'][:]
# timestep  = 0
# replace_every = 8
# x_list = []
# y_list = []
# for i in range(preds.shape[0]):
#     traj =preds[i]
#     y = []
#     x = []
#     for j in range(replace_every):
#         y.append(traj[j, 0])
#         x.append(timestep)
#         timestep += 1
#     x_list.append(x)
#     y_list.append(y)
#
# color = iter(cm.rainbow(np.linspace(0, 1, len(y_list))))
# for i, traj in enumerate(y_list):
#     # c = next(color)
#     plt.plot(x_list[i], traj)
#
# plt.ylabel("Joint command (rads)")
# plt.xlabel("Timestep")
# plt.show()
# print()



### Traj Replacement + Smoothing

preds = demo['actions'][:]
timestep  = 0
replace_every = 8
x_list = []
y_list = []
for i in range(preds.shape[0]):
    traj =preds[i]
    y = []
    x = []
    for j in range(replace_every):
        if j < 3:
            continue
        y.append(traj[j, 0])
        x.append(timestep)
        timestep += 1
    x_list.append(x)
    y_list.append(y)

color = iter(cm.rainbow(np.linspace(0, 1, len(y_list))))
for i, traj in enumerate(y_list):
    # c = next(color)
    plt.plot(x_list[i], traj)

plt.ylabel("Joint command (rads)")
plt.xlabel("Timestep")
plt.show()
print()

