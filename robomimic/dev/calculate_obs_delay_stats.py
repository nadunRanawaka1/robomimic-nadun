import pickle



log_fn = "/home/robot-aiml/ac_learning_repos/experiment_logs/pick_cube_gripper/1x_large_delay.pkl"

with open(log_fn, "rb") as f:
    log = pickle.load(f)

demo = log['demo_0']

joint_msg_times = demo['joint_msg_times']
img_msg_times = demo['img_msg_times']

max_time_delay = []

for i, joint_time in enumerate(joint_msg_times):
    img_time = img_msg_times[i]
    left_time = img_time['sideview_left_camera_rgb']
    right_time = img_time['sideview_right_camera_rgb']
    max_time_delay.append(max(joint_time - left_time, joint_time - right_time))

avg_delay = sum(max_time_delay)/len(max_time_delay)
min_time_delay = min(max_time_delay)
max_delay = max(max_time_delay)
print(f"Avg time delay: {avg_delay}")