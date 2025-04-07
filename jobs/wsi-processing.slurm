#!/bin/bash
#SBATCH --job-name=wsi_processing
#SBATCH --account=gisr97469
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=logs/wsi_processing%j.out

# Load environment
module load 2022
module load 2023
module load "Python/3.10.4-GCCcore-11.3.0"

# Activate your virtual environment
source /projects/0/prjs1477/SG-MuRCL/venv/.clam/bin/activate

# Run patch extraction with segmentation and stitching
python CLAM/create_patches_fp.py \
    --source /projects/0/prjs1477/SG-MuRCL/data/CAMELYON16/images \
    --save_dir /projects/0/prjs1477/SG-MuRCL/data/CAMELYON16-CLAM \
    --patch_size 256 \
    --preset bwh_biopsy.csv \
    --seg \
    --patch \
    --stitch
