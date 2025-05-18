#!/bin/bash
#SBATCH --job-name=clam-patch
#SBATCH --account=gisr97469
#SBATCH --partition=gpu_a100           # or use 'gpu_h100' if you want to try H100s
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=logs/clam-patch%j.out

cd $SLURM_SUBMIT_DIR

# Load modules
module load 2022
module load Python/3.10.4-GCCcore-11.3.0

# Activate your UV venv
source /projects/0/prjs1477/SG-MuRCL/venv/.clam/bin/activate

# Optional: check which GPUs you got
nvidia-smi

DATA="train"

# Run CLAM patch creation
python CLAM/create_patches_fp.py \
    --source /projects/0/prjs1477/SG-MuRCL/data/CAMELYON16/images/${DATA} \
    --save_dir /projects/0/prjs1477/SG-MuRCL/data/C16-CLAM/${DATA} \
    --patch_size 256 \
    --preset /projects/0/prjs1477/SG-MuRCL/CLAM/presets/bwh_biopsy.csv \
    --seg \
    --patch \
    --stitch
