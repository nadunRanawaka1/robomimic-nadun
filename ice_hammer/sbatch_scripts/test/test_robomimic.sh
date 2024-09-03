#!/bin/bash
#SBATCH --job-name robomimic_exps_templates_bc
#SBATCH --output /gv1/projects/ATRP_ACL/logs/robomimic_exps_templates_bc.out
#SBATCH --error /gv1/projects/ATRP_ACL/logs/robomimic_exps_templates_bc.err
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
python scripts/train.py --config="/home/yhe456/src/robomimic-nadun/robomimic/exps/templates/bc.json" --dataset="/gv1/projects/ATRP_ACL/datasets/low_dim_v141.hdf5"
