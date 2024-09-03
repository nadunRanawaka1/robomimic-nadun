#!/bin/bash
#SBATCH --job-name tool_hang_image_diffusion_policy_joint_actions
#SBATCH --output /gv1/projects/ATRP_ACL/logs/tool_hang_image_diffusion_policy_joint_actions.out
#SBATCH --error /gv1/projects/ATRP_ACL/logs/tool_hang_image_diffusion_policy_joint_actions.err
#SBATCH --time 8-00:00
#SBATCH -c 32
#SBATCH --mem=320G
#SBATCH --gres gpu:1
#SBATCH -C "TeslaV100S-PCIE-32GB"

module load gcc/9.2.0
module load anaconda3/2022.05
# Activate environment
conda activate /home/yhe456/.conda/envs/robomimic3

# Change to project directory
cd /home/yhe456/src/robomimic-nadun/robomimic

# Run job
python dev/train.py --config="/home/yhe456/src/robomimic-nadun/ice_hammer/configs/diffusion-policy/sim/joint_model_configs/tool_hang_image_diffusion_policy_joint_actions.json"
