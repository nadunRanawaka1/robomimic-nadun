from os import replace

import nexusformat.nexus as nx
import h5py
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
import numpy as np
import pickle
from bisect import bisect_left

from networkx.algorithms.bipartite import color


def ros_time_to_float(ros_time):
    seconds = ros_time.sec
    nanoseconds = ros_time.nanosec
    float_time = seconds + nanoseconds/(1e+9)
    return float_time

# demo_fn = "/media/nadun/Data/phd_project/robomimic/bc_trained_models/real_robot/logs/rollout_joint_position_single.hdf5"
# demo_fn = "/media/nadun/Data/phd_project/robomimic/bc_trained_models/real_robot/logs/rollout_joint_position_trajectory_with_demo_obs.hdf5"
log_fn = "/media/nadun/Data/phd_project/robomimic/bc_trained_models/real_robot/logs/rollout_temporal_ensemble_1x.pkl"
log_fn = "/media/nadun/Data/phd_project/robomimic/bc_trained_models/real_robot/logs/rollout_no_traj_replacement_1x_speed_no_skipping_large_delay.pkl"

with open(log_fn, "rb") as f:
    log = pickle.load(f)

print()

demo = log['demo_0']
#### PROCESSING THE ACTUAL JOINT POSITIONS
joint_msg_times = []
for msg in demo['all_joint_msg']:
    joint_msg_times.append(ros_time_to_float(msg.header.stamp))

traj_publish_start = demo['traj_publish_times'][0]
traj_publish_start = ros_time_to_float(traj_publish_start)
traj_publish_end =ros_time_to_float(demo['traj_publish_times'][-1])

j_msg_start_ind = bisect_left(joint_msg_times, traj_publish_start)
j_msg_end_ind = bisect_left(joint_msg_times, traj_publish_end) + 100



pt_time = demo['traj_point_time']
actual_actions = demo['actions']
pred_actions = demo['pred_actions']
joint = 0

### CREATE THE DATA FOR actual actions
actual_x_list = []
actual_y_list = []
pred_x_list = []
pred_y_list = []

for i, time in enumerate(demo['traj_publish_times']):
    time = ros_time_to_float(time)
    actual_x = []
    actual_y = []
    pred_y = []
    pred_x = []
    for j in range(16):
        pred_y.append(pred_actions[i, j, joint] * 180/np.pi)
        pred_x.append(time + (j + 1) * pt_time)
        # if j >= demo["skip_joint_actions"]:
        actual_x.append(time + (j + 1) * pt_time)
        actual_y.append(actual_actions[i, j, joint] * 180/np.pi)

    actual_x_list.append(actual_x)
    actual_y_list.append(actual_y)
    pred_y_list.append(pred_y)
    pred_x_list.append(pred_x)


# for i, X in enumerate(actual_x_list):
#     plt.plot(X, actual_y_list[i], color="green", label="temporal ensemble")

for i, X in enumerate(pred_x_list):
    plt.plot(X, pred_y_list[i], color="red", label="predictions")



### Now plot the actual joint states

joint_msgs = demo["all_joint_msg"][j_msg_start_ind:j_msg_end_ind]
j_msg_X = []
j_msg_y = []

for msg in joint_msgs:
    j_msg_X.append(ros_time_to_float(msg.header.stamp))
    j_msg_y.append(msg.position[1] * 180/np.pi)

# plt.plot(j_msg_X, j_msg_y, color="blue", label="actual joint positions")

print()

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
#         if j < 3:
#             continue
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
plt.ylabel("Joint Position")
plt.xlabel("Timestep")
# plt.legend()



handles, labels = plt.gca().get_legend_handles_labels()
# Remove duplicates by converting to a dictionary (which removes duplicates by key)
by_label = dict(zip(labels, handles))
# Create the legend with unique labels
plt.legend(by_label.values(), by_label.keys())
plt.show()
print()

