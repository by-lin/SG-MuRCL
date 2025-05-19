#!/bin/sh

python utils/murclcsv.py \
    --feature_dir data/C16-CLAM/train/features/npz_files \
    --cluster_dir data/C16-CLAM/train/features/npz_files/k-means-10 \
    --output_dir data/C16-CLAM/train \
    --output_name C16_train \
    --reference_csv data/CAMELYON16/evaluation/reference.csv


# Dont forget that 10 is necessary for the MuRCL pretraining
python utils/datasplit.py \
    --train_input data/C16-CLAM/train/C16_train.csv \
    --test_input data/C16-CLAM/test/C16_test.csv \
    --output_dir data/C16-CLAM \
    --output_name C16_split_10