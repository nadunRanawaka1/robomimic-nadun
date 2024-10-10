#!/bin/bash
#SBATCH --job-name=tool_hang_add_all_actions
#SBATCH --output=/coc/flash7/nkra3/logs/sbatch_out/phd_project/processing/tool_hang_add_all_actions.out
#SBATCH --error=/coc/flash7/nkra3/logs/sbatch_err/phd_project/processing/tool_hang_add_all_actions.err
#SBATCH --partition=overcap
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=6
#SBATCH --gpus-per-node="a40:1"
#SBATCH --exclude="clippy"
#SBATCH --exclude="chappie"
#SBATCH --mem-per-gpu=80G

export PYTHONUNBUFFERED=TRUE
source ~/.bashrc
source /nethome/nkra3/flash7/miniconda3/etc/profile.d/conda.sh
conda deactivate
conda activate robomimic-dev


cd /nethome/nkra3/flash7/phd_project/robomimic-nadun/robomimic

srun -u python -u dev/add_all_actions.py --dataset=/nethome/nkra3/flash7/phd_project/robomimic-nadun/datasets/square/ph/all_obs_v141.hdf5