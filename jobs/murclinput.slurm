#!/bin/bash
#SBATCH --job-name=input-murcl
#SBATCH --account=gisr97469
#SBATCH --output=logs/murclcsv_%j.out
#SBATCH --account=gisr97469
#SBATCH --partition=gpu_a100              # or gpu_h100 if needed
#SBATCH --gres=gpu:4                      # 4 GPUs for parallel processing WSIs
#SBATCH --cpus-per-task=8                 # 8 CPUs for multi-threaded pre-processing
#SBATCH --mem=32G                         # RAM for slide loading + patching
#SBATCH --time=12:00:00                    # Walltime 

DATA="test"
# Load env
module purge
module load 2022
module load Python/3.10.4-GCCcore-11.3.0

source /projects/0/prjs1477/SG-MuRCL/venv/.murcl/bin/activate

echo "[$(date)] Starting MuRCL CSV generation on $SLURM_JOB_NODELIST"

python utils/murclcsv.py \
    --feature_dir   /projects/0/prjs1477/SG-MuRCL/data/CAMELYON16-MuRCL/${DATA}/features/resnet50 \
    --cluster_dir   /projects/0/prjs1477/SG-MuRCL/data/CAMELYON16-MuRCL/${DATA}/features/resnet50/k-means-10 \
    --output_dir    /projects/0/prjs1477/SG-MuRCL/data/CAMELYON16-MuRCL/${DATA} \
    --output_name   C16_${DATA} \
    --reference_csv /projects/0/prjs1477/SG-MuRCL/data/CAMELYON16/evaluation/reference.csv

echo "[$(date)] Done."
