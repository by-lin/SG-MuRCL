#!/bin/bash
#SBATCH --job-name=datasplit
#SBATCH --account=gisr97469
#SBATCH --partition=gpu_a100              # or gpu_h100 if needed
#SBATCH --gres=gpu:4                      # 4 GPUs for parallel processing WSIs
#SBATCH --cpus-per-task=8                 # 8 CPUs for multi-threaded pre-processing
#SBATCH --mem=32G                         # RAM for slide loading + patching
#SBATCH --time=12:00:00                    # Walltime 
#SBATCH --output=logs/datasplit_%j.out

# Load modules (if needed)
module purge
module load 2022
module load Python/3.10.4-GCCcore-11.3.0

# Activate your venv if needed
source /projects/0/prjs1477/SG-MuRCL/venv/.clam/bin/activate


# Run the script
python utils/datasplit.py \
    --train_input /projects/0/prjs1477/SG-MuRCL/data/CAMELYON16-MuRCL/train/C16_train.csv \
    --test_input /projects/0/prjs1477/SG-MuRCL/data/CAMELYON16-MuRCL/test/C16_test.csv \
    --output_dir /projects/0/prjs1477/SG-MuRCL/data/CAMELYON16-MuRCL \
    --output_name C16_split_10