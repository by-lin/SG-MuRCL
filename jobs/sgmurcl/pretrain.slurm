#!/bin/bash

#SBATCH --job-name=sgmurcl-pretrain          # job name on queue
#SBATCH --account=gisr97469
#SBATCH --partition=gpu_a100               # queue with A100s
#SBATCH --gres=gpu:a100:4                  # 4 × A100
#SBATCH --cpus-per-task=16                 # host CPU threads visible to the job
#SBATCH --mem=256G                          # system RAM (not GPU RAM)
#SBATCH --time=48:00:00                    # wall-clock limit
#SBATCH --output=logs/sgmurcl-pretrain18-%j.out

echo "========== Node & GPU layout =========="
nvidia-smi
echo "========================================"

module purge
module load 2022
module load Python/3.10.4-GCCcore-11.3.0

# Activate your venv / conda env
source /projects/0/prjs1477/SG-MuRCL/venv/.murcl/bin/activate

bash /projects/0/prjs1477/SG-MuRCL/scripts/pretrain.sh
