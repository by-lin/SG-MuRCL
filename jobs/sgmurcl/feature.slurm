#!/bin/bash
#SBATCH --job-name=sgmurcl-feature
#SBATCH --account=gisr97469
#SBATCH --partition=gpu_a100              # or gpu_h100 if needed
#SBATCH --gres=gpu:4                      # 4 GPUs for parallel processing WSIs
#SBATCH --cpus-per-task=8                 # 8 CPUs for multi-threaded pre-processing
#SBATCH --mem=8G                         # RAM for slide loading + patching
#SBATCH --time=06:00:00                   # Walltime 10 hours for training 270 slides 5 hours for test 129 slides
#SBATCH --output=logs/sgmurcl-feature18sim-test%j.out  # Store stdout + stderr in logs/

# Always work from the project root
cd $SLURM_SUBMIT_DIR
# Load env
module purge
module load 2022
module load Python/3.10.4-GCCcore-11.3.0

source /projects/0/prjs1477/SG-MuRCL/venv/.murcl/bin/activate

# Run patch creation with proper imports
python -m MuRCL.wsi_processing.extract_features \
    --patch_dir /projects/0/prjs1477/SG-MuRCL/data/C16-SGMuRCL/test \
    --image_encoder resnet18 \
    --encoder_weights /projects/0/prjs1477/SG-MuRCL/weights/c16-resnet18-x20.pth
    --device 0,1,2,3

