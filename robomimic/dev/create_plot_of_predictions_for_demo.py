import h5py

import torch

import robomimic
import robomimic.utils.file_utils as FileUtils
import robomimic.utils.env_utils as EnvUtils
import robomimic.utils.torch_utils as TorchUtils
import robomimic.utils.tensor_utils as TensorUtils
import robomimic.utils.obs_utils as ObsUtils

from dev_utils import demo_obs_to_obs_dict



kwargs = {"return_action_sequence": True, "step_action_sequence": True,
              "control_mode": "Joint_Position_Trajectory", "delta_model": False,
              "temporal_ensemble": True, "spline": False, "inpaint_first_action": False,
              "diffusion_sample_n": 10, "return_all_pred": True}


### SETUP THE MODEL
ckpt_path = "/home/robot-aiml/ac_learning_repos/robomimic-nadun/bc_trained_models/real_robot/speed_up_experiments/pick_sube/pick_cube_joint_actions_with_gripper/20240828223719/models/model_epoch_1000.pth"

# device
device = TorchUtils.get_torch_device(try_to_use_cuda=True)

# restore policy
_, ckpt_dict = FileUtils.policy_from_checkpoint(ckpt_path=ckpt_path, device=device, verbose=True)
policy, _ = FileUtils.policy_from_checkpoint(ckpt_path=ckpt_path, device=device, verbose=True)


### SETUP THE DEMO:

demo_fn = "/home/robot-aiml/ac_learning_repos/Task_Demos/merged/move_to_skynet/demo_pick_cube.hdf5"
demo_file = h5py.File(demo_fn)['data']

demo = demo_file['demo_1']
demo_obs = demo['obs']

demo_length = demo.attrs['num_samples']

# Start 'rollout'
policy.start_episode()

for i in range(demo_length):
    obs = demo_obs_to_obs_dict(demo_obs, i)
    act, all_preds = policy(ob=obs, **kwargs)
    print()



print()