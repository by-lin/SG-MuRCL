#!/bin/bash
#SBATCH --job-name=feature_extraction
#SBATCH --account=gisr97469
#SBATCH --partition=gpu_a100           # or use 'gpu_h100' if you want to try H100s
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --output=logs/feature_extraction%j.out

# Load modules
module load 2022
module load Python/3.10.4-GCCcore-11.3.0

# Activate your UV venv
source /projects/0/prjs1477/SG-MuRCL/venv/.clam/bin/activate

# Optional: check which GPUs you got
nvidia-smi

# Run CLAM feature extraction
python CLAM/extract_features_fp.py \
    --data_h5_dir /projects/0/prjs1477/SG-MuRCL/data/CAMELYON16-CLAM \
    --data_slide_dir /projects/0/prjs1477/SG-MuRCL/data/CAMELYON16/images \
    --csv_path /projects/0/prjs1477/SG-MuRCL/data/CAMELYON16-CLAM/process_list_autogen.csv \
    --feat_dir /projects/0/prjs1477/SG-MuRCL/data/CAMELYON16-features \
    --batch_size 512 \
    --slide_ext .tif
