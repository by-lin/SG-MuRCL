#!/bin/bash
################## Slurm directives ##################
#SBATCH --job-name=murcl-pretrain          # job name on queue
#SBATCH --account=gisr97469
#SBATCH --partition=gpu_a100               # queue with A100s
#SBATCH --gres=gpu:a100:4                  # 4 × A100
#SBATCH --cpus-per-task=16                 # host CPU threads visible to the job
#SBATCH --mem=64G                          # system RAM (not GPU RAM)
#SBATCH --time=24:00:00                    # wall-clock limit
#SBATCH --output=logs/murcl-pretrain-clamtest-%j.out
######################################################

# Always start in the directory you submitted from
cd "${SLURM_SUBMIT_DIR}"

echo "========== Node & GPU layout =========="
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
echo "========================================"

################## Load your environment ############
module purge
module load 2022
module load Python/3.10.4-GCCcore-11.3.0   # or whatever module set you use

# Activate your venv / conda env
source /projects/0/prjs1477/SG-MuRCL/venv/.murcl/bin/activate


################## Launch the training ##############
echo "Running MuRCL pre-training (ABMIL, 4×A100)…"
bash MuRCL/runs/pretrain.sh
#####################################################
