#!/bin/bash
#SBATCH --job-name=murcl-clustering
#SBATCH --account=gisr97469
#SBATCH --partition=gpu_a100              # or gpu_h100 if needed
#SBATCH --gres=gpu:4                      # 4 GPUs for parallel processing WSIs
#SBATCH --cpus-per-task=8                 # 8 CPUs for multi-threaded pre-processing
#SBATCH --mem=32G                         # RAM for slide loading + patching
#SBATCH --time=12:00:00                    # Walltime 
#SBATCH --output=logs/og-murcl-clustering-test%j.out  # Store stdout + stderr in logs/

# Always work from the project root
cd $SLURM_SUBMIT_DIR

# Load env
module purge
module load 2022
module load Python/3.10.4-GCCcore-11.3.0

source /projects/0/prjs1477/SG-MuRCL/venv/.murcl/bin/activate
#    --feat_dir /projects/0/prjs1477/SG-MuRCL/data/C16-CLAM/train/features/npz_files \
python -m MuRCL.wsi_processing.features_clustering \
    --feat_dir /projects/0/prjs1477/SG-MuRCL/data/CAMELYON16-MuRCL/test/features/resnet50 \
    --num_cluster 10