#!/bin/bash
#SBATCH --job-name=datasplit
#SBATCH --account=gisr97469
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:10:00
#SBATCH --output=logs/datasplit_%j.out

# Load modules (if needed)
module load 2022
module load Python/3.10.4-GCCcore-11.3.0

# Activate your venv if needed
source /projects/0/prjs1477/SG-MuRCL/venv/.clam/bin/activate

# Set variables
TRAIN_CSV="/projects/0/prjs1477/SG-MuRCL/data/C16-CLAM/train/train.csv"
TEST_CSV="/projects/0/prjs1477/SG-MuRCL/data/C16-CLAM/test/test.csv"
OUTDIR="/projects/0/prjs1477/SG-MuRCL/data/C16-CLAM/dev/"
OUTNAME="split"
VALRATIO=0.2
SEED=985

# Run the script
python utils/datasplit.py \
    --train_input "$TRAIN_CSV" \
    --test_input "$TEST_CSV" \
    --output_dir "$OUTDIR" \
    --output_name "$OUTNAME" \
    --val_ratio $VALRATIO \
    --random_seed $SEED