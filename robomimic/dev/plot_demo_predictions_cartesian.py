import h5py

import torch

import robomimic
import robomimic.utils.file_utils as FileUtils
import robomimic.utils.env_utils as EnvUtils
import robomimic.utils.torch_utils as TorchUtils
import robomimic.utils.tensor_utils as TensorUtils
import robomimic.utils.obs_utils as ObsUtils

from dev_utils import demo_obs_to_obs_dict, postprocess_obs
import matplotlib.pyplot as plt


### CONSTANTS

kwargs = {"return_action_sequence": True, "step_action_sequence": True,
              "diffusion_sample_n": 10, "return_all_pred": True}


POINT_TIME = 0.05
PLOT_AXIS = 0
RUN_ACTIONS = 2




### SETUP THE MODEL
ckpt_path = "/home/robot-aiml/ac_learning_repos/robomimic-nadun/bc_trained_models/real_robot/speed_up_experiments/pick_cube_realsense/pick_cube_ee_control_framestack_2/20240905232039/models/model_epoch_1200.pth"

# device
device = TorchUtils.get_torch_device(try_to_use_cuda=True)

# restore policy
_, ckpt_dict = FileUtils.policy_from_checkpoint(ckpt_path=ckpt_path, device=device, verbose=True)
policy, _ = FileUtils.policy_from_checkpoint(ckpt_path=ckpt_path, device=device, verbose=True)


### SETUP THE DEMO:

demo_fn = "/home/robot-aiml/ac_learning_repos/Task_Demos/merged/demo_pick_cube_realsense_holdout.hdf5"
# demo_fn = "/home/robot-aiml/ac_learning_repos/Task_Demos/merged/move_to_skynet/demo_pick_cube_realsense.hdf5"
demo_file = h5py.File(demo_fn)['data']

demo = demo_file['demo_2']
demo_obs = demo['obs']
demo_actions = demo['absolute_axis_angle_actions']

demo_length = demo.attrs['num_samples']

# Start 'rollout'
policy.start_episode()



pred_actions = []
obs_ee_pos = []
prev_actions_completed = []
drop_action_list = []
all_preds_list = []

obs_x_list = []
obs_y_list = []
pred_x_list = []
pred_y_list = []
demo_acts = []

for i in range(0, demo_length, RUN_ACTIONS):
    obs = demo_obs_to_obs_dict(demo_obs, i)
    obs = postprocess_obs(obs)
    act, all_preds = policy(ob=obs, **kwargs)

    obs_ee_pos.append(obs['ee_pose'][PLOT_AXIS])
    pred_actions.append(act)
    all_preds_list.append(all_preds)

    # Generating stuff for plotting
    obs_x_list.append([i*POINT_TIME])
    obs_y_list.append([obs['ee_pose'][PLOT_AXIS]])

    demo_acts.append(demo_actions[i][PLOT_AXIS])

    pred_x = []
    pred_y = []
    for j in range(act.shape[0]):
        pred_x.append((i*POINT_TIME) + (j + 1)*POINT_TIME)
        pred_y.append(act[j, PLOT_AXIS])

    pred_x_list.append(pred_x)
    pred_y_list.append(pred_y)

for i, X in enumerate(pred_x_list):
    # plt.plot(obs_x_list[i], obs_y_list[i], label="x position at time of prediction", ls='None', marker=f'${str(i)}$',
    #          color='red', markersize=18)
    plt.plot(obs_x_list[i], demo_acts[i], label="commanded x position in demo", ls='None', marker=f'${str(i)}$',
             color='red', markersize=18)
    plt.plot(X, pred_y_list[i], label="predictions", ls='None', marker=f'${str(i)}$', color="orange", markersize=14)


plt.ylabel("X Position of robot")
plt.xlabel("Timestep")


plt.title("Rollout with Demo Obs")
handles, labels = plt.gca().get_legend_handles_labels()
# Remove duplicates by converting to a dictionary (which removes duplicates by key)
by_label = dict(zip(labels, handles))
# Create the legend with unique labels
plt.legend(by_label.values(), by_label.keys())
plt.show()
