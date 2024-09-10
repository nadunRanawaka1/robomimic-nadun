import matplotlib.pyplot as plt
import numpy as np
import h5py

base_path = "bc_trained_models/diffusion_policy/can_image_diffusion_policy_joint_actions/20240904041148/"
trajectory_steps = 8
trajectory_timestep = 0.05
SIMULATION_TIMESTEP = 0.002
kp = 50



# Controller Joint Trajectory
controller_obs = h5py.File(base_path+f"rollout_obs/joint_traj_controller_records_{trajectory_timestep}.hdf5", "r")
controller_time = controller_obs["controller_step"][:] * SIMULATION_TIMESTEP
traj_goal = controller_obs["traj_goal"]
goal_qpos = controller_obs["goal_qpos"]
joint_pos = controller_obs["joint_pos"]
joint0_traj_goal = traj_goal[:,0]
joint0_goal_qpos = goal_qpos[:,0]
joint0_joint_pos = joint_pos[:,0]

torque = controller_obs["torque"]
desired_torque = controller_obs["desired_torque"]
joint0_torque = torque[:,0]
joint0_desired_torque = desired_torque[:,0]/kp
position_error = controller_obs["position_error"]
joint0_position_error = position_error[:,0]

plt.plot(controller_time, joint0_traj_goal, 'o--', label="joint 0 trajectory goal")
plt.plot(controller_time, joint0_goal_qpos, 'x--', label="joint 0 goal")
plt.plot(controller_time, joint0_joint_pos, '+--', label="joint 0 pos")
# plt.plot(controller_time, joint0_torque, 'x--', label="joint 0 torque")
# plt.plot(controller_time, joint0_desired_torque, 'o--', label="joint 0 desired torque")
# plt.plot(controller_time, joint0_position_error, '^--', label="joint 0 position error")

# Observation
# obs = h5py.File(base_path+f"rollout_obs/joint_traj_timestep_{trajectory_timestep}.hdf5", "r")
# action = obs["data"]["demo_0"]["actions"]
# obs = obs["data"]["demo_0"]["next_obs"]
# sim_time = obs["cur_time"]
# robot_joint_pos = obs["robot0_joint_pos"]
# time = sim_time[:,1,:]
# times = np.tile(time, (1, trajectory_steps))
# for i in range(trajectory_steps):
#     times[:,i] -= trajectory_timestep*(trajectory_steps - 1 - i)

# joint0_action = action[:, :,0]
# joint0_pos = robot_joint_pos[:,-1,0]

# plt.plot(times, joint0_action, 'o--', label="joint 0 actions")
# plt.plot(time, joint0_pos, 'x--', label="joint 0 pos")

plt.title("Robot Joint Trajectory")
plt.xlabel("Time (s)")
plt.ylabel("Joint Positon (rad)")
# plt.ylabel("Torques")
plt.legend()
plt.grid(True)

plt.show()