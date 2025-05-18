#!/bin/sh

python utils/murclcsv.py \
    --feature_dir data/C16-CLAM/dev/features/npz_files \
    --cluster_dir data/C16-CLAM/dev/features/npz_files/k-means-10 \
    --output_dir data/C16-CLAM/dev


# Dont forget that 10 is necessary for the MuRCL pretraining
python utils/datasplit.py \
    --train_input data/C16-CLAM/dev/input.csv \
    --test_input data/C16-CLAM/dev/input.csv \
    --output_dir data/C16-CLAM \
    --output_name traintest_split_10.json \