#!/bin/bash
#SBATCH --job-name=murcl-patch
#SBATCH --account=gisr97469
#SBATCH --partition=gpu_a100              # or gpu_h100 if needed
#SBATCH --gres=gpu:4                      # 4 GPUs for parallel processing WSIs
#SBATCH --cpus-per-task=8                 # 8 CPUs for multi-threaded pre-processing
#SBATCH --mem=16G                         # RAM for slide loading + patching
#SBATCH --time=08:00:00                   # Walltime
#SBATCH --output=logs/murcl-patch-%j.out  # Store stdout + stderr in logs/

# Always work from the project root
cd $SLURM_SUBMIT_DIR

# Load env
module purge
module load 2022
module load Python/3.10.4-GCCcore-11.3.0

source /projects/0/prjs1477/SG-MuRCL/venv/.murcl/bin/activate

# Run patch creation with proper imports
python -m MuRCL.wsi_processing.create_patches \
    --slide_dir /projects/0/prjs1477/SG-MuRCL/data/CAMELYON16/train \
    --save_dir /projects/0/prjs1477/SG-MuRCL/data/CAMELYON16-MuRCL/train \
    --overview \
    --save_mask \
    --wsi_format .tif \
    --overview_level 5