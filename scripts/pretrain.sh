#!/bin/sh

ENCODER=resnet50
DATASET=C16-SGMuRCL

# Stage 1: Warm-up Training (Contrastive Learning)
python run/train_MuRCL.py \
  --dataset CAMELYON16 \
  --data_csv /projects/0/prjs1477/SG-MuRCL/data/${DATASET}/${ENCODER}_input_with_clusters_10.csv \
  --data_split_json /projects/0/prjs1477/SG-MuRCL/data/${DATASET}/c16_split_10.json \
  --feat_size 1024 \
  --preload \
  --train_stage 1 \
  --T 6 \
  --scheduler CosineAnnealingLR \
  --batch_size 64 \
  --epochs 100 \
  --backbone_lr 0.0001 \
  --fc_lr 0.00001 \
  --wdecay 1e-5 \
  --patience 10 \
  --device 0,1,2,3 \
  --save_dir_flag C16-SG50-PRETRAIN \
  --graph_encoder_type gat \
  --mil_aggregator_type smtabmil \
  --graph_level patch \
  --num_clusters 10 \
  --gnn_hidden_dim 256 \
  --gnn_output_dim 256 \
  --gnn_num_layers 2 \
  --gat_heads 4 \
  --gnn_dropout 0.1 \
  --gnn_lr 0.001 \
  --gnn_weight_decay 5e-4 \
  --mil_lr 0.0001 \
  --mil_weight_decay 1e-4 \
  --model_dim 512 \
  --projection_dim 128 \
  --D 128 \
  --dropout 0.1 \
  --temperature 1.0 \
  --alpha 0.9 \
  --exist_ok

# Stage 2: PPO Training (Reinforcement Learning)
python run/train_MuRCL.py \
  --dataset CAMELYON16 \
  --data_csv /projects/0/prjs1477/SG-MuRCL/data/${DATASET}/${ENCODER}_input_with_clusters_10.csv \
  --data_split_json /projects/0/prjs1477/SG-MuRCL/data/${DATASET}/c16_split_10.json \
  --feat_size 1024 \
  --preload \
  --train_stage 2 \
  --T 6 \
  --scheduler CosineAnnealingLR \
  --batch_size 64 \
  --epochs 30 \
  --backbone_lr 0.00001 \
  --fc_lr 0.00001 \
  --wdecay 1e-5 \
  --patience 10 \
  --device 0,1,2,3 \
  --save_dir_flag C16-SG50-PRETRAIN \
  --graph_encoder_type gat \
  --mil_aggregator_type smtabmil \
  --graph_level patch \
  --num_clusters 10 \
  --gnn_hidden_dim 256 \
  --gnn_output_dim 256 \
  --gnn_num_layers 2 \
  --gat_heads 4 \
  --gnn_dropout 0.1 \
  --gnn_lr 0.001 \
  --gnn_weight_decay 5e-4 \
  --mil_lr 0.0001 \
  --mil_weight_decay 1e-4 \
  --model_dim 512 \
  --projection_dim 128 \
  --D 128 \
  --dropout 0.1 \
  --action_std 0.5 \
  --ppo_lr 0.00001 \
  --ppo_gamma 0.1 \
  --K_epochs 3 \
  --temperature 1.0 \
  --alpha 0.9 \
  --exist_ok

# Stage 3: Fine-tuning (End-to-End)
python run/train_MuRCL.py \
  --dataset CAMELYON16 \
  --data_csv /projects/0/prjs1477/SG-MuRCL/data/${DATASET}/${ENCODER}_input_with_clusters_10.csv \
  --data_split_json /projects/0/prjs1477/SG-MuRCL/data/${DATASET}/c16_split_10.json \
  --feat_size 1024 \
  --preload \
  --train_stage 3 \
  --T 6 \
  --scheduler CosineAnnealingLR \
  --batch_size 64 \
  --epochs 100 \
  --backbone_lr 0.00005 \
  --fc_lr 0.00001 \
  --wdecay 1e-5 \
  --patience 10 \
  --device 0,1,2,3 \
  --save_dir_flag C16-50-PRETRAIN \
  --graph_encoder_type gat \
  --mil_aggregator_type smtabmil \
  --graph_level patch \
  --num_clusters 10 \
  --gnn_hidden_dim 256 \
  --gnn_output_dim 256 \
  --gnn_num_layers 2 \
  --gat_heads 4 \
  --gnn_dropout 0.1 \
  --gnn_lr 0.001 \
  --gnn_weight_decay 5e-4 \
  --mil_lr 0.0001 \
  --mil_weight_decay 1e-4 \
  --model_dim 512 \
  --projection_dim 128 \
  --D 128 \
  --dropout 0.1 \
  --action_std 0.5 \
  --ppo_lr 0.00001 \
  --ppo_gamma 0.1 \
  --K_epochs 3 \
  --temperature 1.0 \
  --alpha 0.9 \
  --exist_ok