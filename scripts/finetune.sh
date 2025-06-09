#!/bin/sh

ENCODER=c16x20-simclr-resnet18
DATASET=C16-SGMuRCL
EXP=exp_C16-BASELINE18SIM-PRETRAIN

# STAGE 1
python MuRCL/train_RLMIL.py \
  --dataset CAMELYON16 \
  --data_csv /projects/0/prjs1477/SG-MuRCL/data/${DATASET}/${ENCODER}_input_10.csv \
  --data_split_json /projects/0/prjs1477/SG-MuRCL/data/${DATASET}/c16_split_10.json \
  --train_data train \
  --feat_size 1024 \
  --preload \
  --train_method finetune \
  --train_stage 1 \
  --checkpoint_pretrained /projects/0/prjs1477/SG-MuRCL/results/CAMELYON16_np_1024/MuRCL/T6_pd128_as0.5_pg0.1_tau1.0_alpha0.9/ABMIL/L512_D128_dpt0.0/${EXP}/seed985/stage_3/model_best.pth.tar \
  --T 6 \
  --scheduler CosineAnnealingLR \
  --batch_size 8 \
  --epochs 100 \
  --backbone_lr 0.0001 \
  --fc_lr 0.00001 \
  --wdecay 1e-5 \
  --patience 10 \
  --arch ABMIL \
  --device 0,1,2,3 \
  --save_model \
  --save_dir_flag C16-BASELINE18SIM-FINETUNE \
  --exist_ok

# STAGE 2
python MuRCL/train_RLMIL.py \
  --dataset CAMELYON16 \
  --data_csv /projects/0/prjs1477/SG-MuRCL/data/${DATASET}/${ENCODER}_input_10.csv \
  --data_split_json /projects/0/prjs1477/SG-MuRCL/data/${DATASET}/c16_split_10.json \
  --train_data train \
  --feat_size 1024 \
  --preload \
  --train_method finetune \
  --train_stage 2 \
  --checkpoint_stage /projects/0/prjs1477/SG-MuRCL/results/CAMELYON16_np_1024/RLMIL/T6_as0.5_pg0.1_phd512_fhd1024/ABMIL/L512_D128_dpt0.0/finetune/exp_C16-BASELINE18SIM-FINETUNE/seed985/stage_1/model_best.pth.tar \
  --checkpoint_pretrained /projects/0/prjs1477/SG-MuRCL/results/CAMELYON16_np_1024/MuRCL/T6_pd128_as0.5_pg0.1_tau1.0_alpha0.9/ABMIL/L512_D128_dpt0.0/${EXP}/seed985/stage_3/model_best.pth.tar \
  --T 6 \
  --scheduler CosineAnnealingLR \
  --batch_size 8 \
  --epochs 30 \
  --backbone_lr 0.00001 \
  --fc_lr 0.00001 \
  --wdecay 1e-5 \
  --patience 10 \
  --arch ABMIL \
  --device 0,1,2,3 \
  --save_model \
  --save_dir_flag C16-BASELINE18SIM-FINETUNE \
  --exist_ok

# STAGE 3
python MuRCL/train_RLMIL.py \
  --dataset CAMELYON16 \
  --data_csv /projects/0/prjs1477/SG-MuRCL/data/${DATASET}/${ENCODER}_input_10.csv \
  --data_split_json /projects/0/prjs1477/SG-MuRCL/data/${DATASET}/c16_split_10.json \
  --train_data train \
  --feat_size 1024 \
  --preload \
  --train_method finetune \
  --train_stage 3 \
  --checkpoint_stage /projects/0/prjs1477/SG-MuRCL/results/CAMELYON16_np_1024/RLMIL/T6_as0.5_pg0.1_phd512_fhd1024/ABMIL/L512_D128_dpt0.0/finetune/exp_C16-BASELINE18SIM-FINETUNE/seed985/stage_2/model_best.pth.tar \
  --T 6 \
  --scheduler CosineAnnealingLR \
  --batch_size 8 \
  --epochs 100 \
  --backbone_lr 0.00005 \
  --fc_lr 0.00001 \
  --wdecay 1e-5 \
  --patience 10 \
  --arch ABMIL \
  --device 0,1,2,3 \
  --save_model \
  --save_dir_flag C16-BASELINE18SIM-FINETUNE \
  --exist_ok