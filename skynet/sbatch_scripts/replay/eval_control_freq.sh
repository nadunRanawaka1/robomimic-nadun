#!/bin/bash
#SBATCH --job-name=can_tracking_error_eval_jp
#SBATCH --output=/coc/flash7/nkra3/logs/sbatch_out/phd_project/replay/can_tracking_error_eval_jp.out
#SBATCH --error=/coc/flash7/nkra3/logs/sbatch_err/phd_project/replay/can_tracking_error_eval_jp.err
#SBATCH --partition=overcap
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=6
#SBATCH --gpus-per-node="titan_x:1"
#SBATCH --exclude="clippy"
#SBATCH --exclude="chappie"
#SBATCH --mem-per-gpu=80G

export PYTHONUNBUFFERED=TRUE
source ~/.bashrc
source /nethome/nkra3/flash7/miniconda3/etc/profile.d/conda.sh
conda deactivate
conda activate robomimic-dev


cd /nethome/nkra3/flash7/phd_project/robomimic-nadun/robomimic

srun -u python -u dev/evaluate_tracking_error.py \
    --dataset="/nethome/nkra3/flash7/phd_project/robomimic-nadun/datasets/can/ph/all_obs_v141.hdf5" \
    --save_path="/nethome/nkra3/flash7/phd_project/experiment_logs/control_freq_replay/can_eval_jp.pkl"

