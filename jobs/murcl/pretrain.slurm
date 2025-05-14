#!/bin/bash
#SBATCH --job-name=murcl-pretrain
#SBATCH --account=gisr97469
#SBATCH --partition=gpu_a100
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=logs/murcl-pretrain-%j.out

# Always work from the project root
cd $SLURM_SUBMIT_DIR

nvdia-smi

# Load environment
module purge
module load 2022
module load Python/3.10.4-GCCcore-11.3.0

source /projects/0/prjs1477/SG-MuRCL/venv/.murcl/bin/activate

# Optional: copy this run to a new experiment folder
echo "Running MuRCL pretraining on 4×A100 with ABMIL backbone"

bash MuRCL/runs/pretrain.sh
