from os import replace

import nexusformat.nexus as nx
import h5py
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
import numpy as np
import pickle
from bisect import bisect_left

from networkx.algorithms.bipartite import color


PLOT_JOINT = 0
JOINT_TO_NAME = {
    0: "elbow_joint",
    1: "shoulder_lift_joint",
    2: "shoulder_pan_joint",
    3: "wrist_1_joint",
    4: "wrist_2_joint",
    5: "wrist_3_joint"
}

ANGLE_MULTIPLIER = 1


def ros_time_to_float(ros_time):
    seconds = ros_time.sec
    nanoseconds = ros_time.nanosec
    float_time = seconds + nanoseconds/(1e+9)
    return float_time

# demo_fn = "/media/nadun/Data/phd_project/robomimic/bc_trained_models/real_robot/logs/rollout_joint_position_single.hdf5"
# demo_fn = "/media/nadun/Data/phd_project/robomimic/bc_trained_models/real_robot/logs/rollout_joint_position_trajectory_with_demo_obs.hdf5"
log_fn = "/media/nadun/Data/phd_project/robomimic/bc_trained_models/real_robot/logs/rollout_temporal_ensemble_1x.pkl"
log_fn = "/home/robot-aiml/ac_learning_repos/robomimic-nadun/bc_trained_models/real_robot/strawberry/strawberry_joint_position_actions/20240808154101/logs/rollout_blended_actions_traj_replacement_half_speed_drop_action_testing.pkl"

log_fn = "/home/robot-aiml/ac_learning_repos/robomimic-nadun/bc_trained_models/real_robot/strawberry/strawberry_joint_position_actions/20240808154101/logs/rollout_blended_actions_traj_replacement_half_speed_drop_action.pkl"
log_fn = "/home/robot-aiml/ac_learning_repos/robomimic-nadun/bc_trained_models/real_robot/strawberry/strawberry_joint_position_actions/20240808154101/logs/rollout_fixed_window_blended_actions_half_speed.pkl"
log_fn = "/home/robot-aiml/ac_learning_repos/robomimic-nadun/bc_trained_models/real_robot/strawberry/strawberry_joint_position_actions/20240808154101/logs/rollout_fixed_window_blended_actions_2x_speed.pkl"

log_fn = "/home/robot-aiml/ac_learning_repos/experiment_logs/pick_cube_gripper/blended_actions_1x_sample_multiple.pkl"

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
obs_joint_pos = demo['obs_joint_pos']
drop_actions = demo['drop_action_list']


### CREATE THE DATA FOR actual actions
actual_x_list = []
actual_y_list = []
pred_x_list = []
pred_y_list = []

obs_x_list = []
obs_y_list = []

for i, time in enumerate(demo['traj_start_times']):
    time = ros_time_to_float(time)
    actual_x = []
    actual_y = []
    pred_y = []
    pred_x = []

    obs_x_list.append(time)
    obs_y_list.append(obs_joint_pos[i][PLOT_JOINT]* ANGLE_MULTIPLIER)
    for j in range(16):
        pred_y.append(pred_actions[i][j, PLOT_JOINT] * ANGLE_MULTIPLIER)
        pred_x.append(time + (j + 1) * pt_time)
    for k in range(actual_actions[i].shape[0]):
        actual_x.append(time + (k + 1 + drop_actions[i]) * pt_time)
        actual_y.append(actual_actions[i][k, PLOT_JOINT] * ANGLE_MULTIPLIER)

    actual_x_list.append(actual_x)
    actual_y_list.append(actual_y)
    pred_y_list.append(pred_y)
    pred_x_list.append(pred_x)


for i, X in enumerate(pred_x_list):
    plt.plot(obs_x_list[i], obs_y_list[i], label="joint state at time of prediction", ls='None', marker=f'${str(i)}$',
             color='red', markersize=16)
    plt.plot(X, pred_y_list[i], label="predictions", ls='None', marker=f'${str(i)}$', color="orange", markersize=14)

#
# for i, X in enumerate(actual_x_list):
#     plt.plot(X, actual_y_list[i], label="blended actions", ls="None", marker=f'${str(i)}$', markersize=10, color="green")



### Now plot the actual joint states

joint_msgs = demo["all_joint_msg"][j_msg_start_ind:j_msg_end_ind]
j_msg_X = []
j_msg_y = []

for msg in joint_msgs:
    joint_name = JOINT_TO_NAME[PLOT_JOINT]
    ind = msg.name.index(joint_name)
    j_msg_X.append(ros_time_to_float(msg.header.stamp))
    j_msg_y.append(msg.position[ind] * ANGLE_MULTIPLIER)

### PLOT THE ACTUAL JOINT STATE
# plt.plot(j_msg_X, j_msg_y, color="blue", label="actual joint positions")

print()

### Plot the actual published traj msgs
traj_msgs = demo['published_traj_msgs']
msg_X_list = []
msg_y_list = []

for msg in traj_msgs:
    msg_X = []
    msg_y = []
    for point in msg.points:
        x = ros_time_to_float(msg.header.stamp) + ros_time_to_float(point.time_from_start)
        msg_X.append(x)
        msg_y.append(point.positions[PLOT_JOINT]* ANGLE_MULTIPLIER)
    msg_X_list.append(msg_X)
    msg_y_list.append(msg_y)

### PLOT THE ACTUAL MSGS
# for i, X in enumerate(msg_X_list):
#     plt.plot(X, msg_y_list[i], color="purple", label="published traj msgs", marker="*", ls="None")

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

