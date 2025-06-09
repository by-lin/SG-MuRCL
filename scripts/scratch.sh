#!/bin/sh

ENCODER=resnet18
DATASET=C16-SGMuRCL

python MuRCL/train_RLMIL.py \
  --dataset CAMELYON16 \
  --data_csv /projects/0/prjs1477/SG-MuRCL/data/${DATASET}/${ENCODER}_input_10.csv \
  --data_split_json /projects/0/prjs1477/SG-MuRCL/data/${DATASET}/${ENCODER}_split_10.json \
  --train_data train \
  --feat_size 1024 \
  --preload \
  --train_method scratch \
  --train_stage 1 \
  --T 6 \
  --scheduler CosineAnnealingLR \
  --batch_size 1 \
  --epochs 40 \
  --backbone_lr 0.0001 \
  --fc_lr 0.00005 \
  --arch ABMIL \
  --device 3 \
  --save_model \
  --exist_ok

python MuRCL/train_RLMIL.py \
  --dataset CAMELYON16 \
  --data_csv /projects/0/prjs1477/SG-MuRCL/data/${DATASET}/${ENCODER}_input_10.csv \
  --data_split_json /projects/0/prjs1477/SG-MuRCL/data/${DATASET}/${ENCODER}_split_10.json \
  --train_data train \
  --feat_size 1024 \
  --preload \
  --train_method scratch \
  --train_stage 2 \
  --T 6 \
  --scheduler CosineAnnealingLR \
  --batch_size 1 \
  --epochs 40 \
  --backbone_lr 0.0001 \
  --fc_lr 0.00005 \
  --arch ABMIL \
  --device 3 \
  --save_model \
  --exist_ok

python MuRCL/train_RLMIL.py \
  --dataset CAMELYON16 \
  --data_csv /projects/0/prjs1477/SG-MuRCL/data/${DATASET}/${ENCODER}_input_10.csv \
  --data_split_json /projects/0/prjs1477/SG-MuRCL/data/${DATASET}/${ENCODER}_split_10.json \
  --train_data train \
  --feat_size 1024 \
  --preload \
  --train_method scratch \
  --train_stage 3 \
  --T 6 \
  --scheduler CosineAnnealingLR \
  --batch_size 1 \
  --epochs 40 \
  --backbone_lr 0.00005 \
  --fc_lr 0.00001 \
  --arch ABMIL \
  --device 3 \
  --save_model \
  --exist_ok