#!/bin/bash
#SBATCH --job-name=lift_reached_osc_control_freq_eval
#SBATCH --output=/coc/flash7/nkra3/logs/sbatch_out/phd_project/eval/lift_reached_osc_control_freq_eval.out
#SBATCH --error=/coc/flash7/nkra3/logs/sbatch_err/phd_project/eval/lift_reached_osc_control_freq_eval.err
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

srun -u python -u dev/run_trained_agent_aggregated_actions.py --agent=/nethome/nkra3/flash7/phd_project/robomimic-nadun/bc_trained_models/diffusion_policy/sim/reached_osc/lift_image/20241010114617/models/model_epoch_600.pth