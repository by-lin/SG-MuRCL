#!/bin/sh

python utils/murclcsv.py \
    --feature_dir data/CAMELYON16-MuRCL/test/features/resnet18 \
    --cluster_dir data/CAMELYON16-MuRCL/test/features/resnet18/k-means-10 \
    --output_dir data/CAMELYON16-MuRCL/test

python utils/datasplit.py \
    --train_input data/CAMELYON16-MuRCL/train/input.csv \
    --test_input data/CAMELYON16-MuRCL/test/input.csv \
    --output_dir data/CAMELYON16-MuRCL