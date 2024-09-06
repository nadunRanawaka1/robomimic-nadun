import h5py

base_path = "bc_trained_models/diffusion_policy/can_image_diffusion_policy_joint_actions/20240904041148/"
trajectory_steps = 8
trajectory_timestep = 0.1

f = h5py.File(base_path+f"rollout_obs/joint_traj_timestep_{trajectory_timestep}.hdf5", "r")
action = f["data"]["demo_0"]["actions"]
obs = f["data"]["demo_0"]["next_obs"]
sim_time = obs["cur_time"]
robot_joint_pos = obs["robot0_joint_pos"]
print()

import matplotlib.pyplot as plt
import numpy as np

time = sim_time[:,1,:]
times = np.tile(time, (1, trajectory_steps))
for i in range(trajectory_steps):
    times[:,i] -= trajectory_timestep*(trajectory_steps - 1 - i)

joint0_action = action[:, :,0]
joint0_pos = robot_joint_pos[:,-1,0]

plt.plot(times, joint0_action, 'o--', label="joint 0 actions")
plt.plot(time, joint0_pos, 'x--', label="joint 0 pos")

plt.title("Robot Joint Trajectory")
plt.xlabel("Time (s)")
plt.ylabel("Joint Positon (rad)")
plt.legend()
plt.grid(True)

plt.show()