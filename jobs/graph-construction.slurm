#!/bin/bash
#SBATCH --job-name=graph_construction
#SBATCH --account=gisr97469
#SBATCH --partition=gpu_a100           # or use 'gpu_h100' if you want to try H100s
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=logs/graph_construction%j.out

# Load modules
module purge
module load 2022
module load Python/3.10.4-GCCcore-11.3.0
module load OpenSlide/3.4.1-GCCcore-11.3.0-largefiles

# Activate your UV venv
source /projects/0/prjs1477/SG-MuRCL/venv/.gmil/bin/activate

# Optional: check which GPUs you got
nvidia-smi

# Run GMIL graph construction
python GMIL/STEP1_build_graphs.py \
    --type_graph patch \
    --h5_path /projects/0/prjs1477/SG-MuRCL/data/CAMELYON16-features/h5_training.csv \
    --feats_path /projects/0/prjs1477/SG-MuRCL/data/CAMELYON16-features/h5_files/training \
    --graph_save_path /projects/0/prjs1477/SG-MuRCL/data/CAMELYON16-graphs \
    --num_classes 1 \
    --num_clusters 9 \
    --patch_size 256 \
    --dataset CAMELYON16
