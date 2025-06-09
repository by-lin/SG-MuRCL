#!/bin/bash

# Se lanza: nohup ./code/bash_scripts/run_camelyon.sh &

experiment_name=run_camelyon

mkdir -p output/${experiment_name}

rm -rf output/${experiment_name}/out_*

export CUDA_DEVICE_ORDER=PCI_BUS_ID

export CUDA_VISIBLE_DEVICES=2

seeds_array=(1 2 3 4 5)

num_workers=1
val_prop=0.2

epochs=50
patience=$epochs
# patience=$((epochs / 5))

dataset=camelyon16-patches_512_preset-features_resnet50_bt
# dataset=camelyon16-patches_512_preset-features_resnet18
# dataset=camelyon16-patches_512_preset-features_resnet50
# dataset=camelyon16-patches_512_preset-features_vit_b_32

config_file=/work/work_fran/SmMIL/code/experiments/config.yml

############################################################################################################

# ABMIL

model_name=abmil
batch_size=1
lr_array=(0.0001)

for seed in "${seeds_array[@]}"
do
    for lr in "${lr_array[@]}"
    do
        python code/experiments/run_experiment.py --mode=train_test --use_wandb --num_workers=$num_workers --seed=$seed \
            --dataset=$dataset --batch_size=$batch_size --epochs=$epochs --val_prop=$val_prop \
            --model_name=$model_name --lr=$lr --patience=$patience \
            > output/${experiment_name}/out_${BASHPID}.txt 2>&1
    done
done

############################################################################################################

# SmABMIL

model_name=sm_abmil
batch_size=1
lr_array=(0.0001)
sm_where_array=(early mid late)

for seed in "${seeds_array[@]}"
do
    for lr in "${lr_array[@]}"
    do
        for sm_where in "${sm_where_array[@]}"
        do
            python code/experiments/run_experiment.py --mode=train_test --use_wandb --num_workers=$num_workers --seed=$seed \
                --dataset=$dataset --batch_size=$batch_size --epochs=$epochs --val_prop=$val_prop \
                --model_name=$model_name --lr=$lr --patience=$patience --sm_where=$sm_where --sm_spectral_norm  --use_sparse --use_inst_distances \
                > output/${experiment_name}/out_${BASHPID}.txt 2>&1
        done
    done
done

###########################################################################################################

# Transformer + ABMIL

model_name=transformer_abmil
batch_size=1
lr_array=(0.0001)

for seed in "${seeds_array[@]}"
do
    for lr in "${lr_array[@]}"
    do
        python code/experiments/run_experiment.py --mode=train_test --use_wandb --num_workers=$num_workers --seed=$seed \
            --dataset=$dataset --batch_size=$batch_size --epochs=$epochs --val_prop=$val_prop \
            --model_name=$model_name --lr=$lr --patience=$patience  --use_sparse  \
            > output/${experiment_name}/out_${BASHPID}.txt 2>&1
    done
done

###########################################################################################################

# SmTransformer + ABMIL

model_name=sm_transformer_abmil
batch_size=1
lr_array=(0.0001)

for seed in "${seeds_array[@]}"
do
    for lr in "${lr_array[@]}"
    do
        python code/experiments/run_experiment.py --mode=train_test --use_wandb --num_workers=$num_workers --seed=$seed \
            --dataset=$dataset --batch_size=$batch_size --epochs=$epochs --val_prop=$val_prop \
            --model_name=$model_name --lr=$lr --patience=$patience --sm_transformer --sm_spectral_norm  --use_sparse --use_inst_distances \
            > output/${experiment_name}/out_${BASHPID}.txt 2>&1
    done
done

###########################################################################################################

# Transformer + SmABMIL

model_name=sm_transformer_abmil
batch_size=1
lr_array=(0.0001)
sm_where_array=(early mid late)

for seed in "${seeds_array[@]}"
do
    for lr in "${lr_array[@]}"
    do
        for sm_where in "${sm_where_array[@]}"
        do
            python code/experiments/run_experiment.py --mode=train_test --use_wandb --num_workers=$num_workers --seed=$seed \
                --dataset=$dataset --batch_size=$batch_size --epochs=$epochs --val_prop=$val_prop \
                --model_name=$model_name --lr=$lr --patience=$patience --sm_where=$sm_where --sm_spectral_norm  --use_sparse --use_inst_distances \
                > output/${experiment_name}/out_${BASHPID}.txt 2>&1
        done
    done
done

###########################################################################################################

# SmTransformer + SmABMIL

model_name=sm_transformer_abmil
batch_size=1
lr_array=(0.0001)
sm_where_array=(early mid late)

for seed in "${seeds_array[@]}"
do
    for lr in "${lr_array[@]}"
    do
        for sm_where in "${sm_where_array[@]}"
        do
            python code/experiments/run_experiment.py --mode=train_test --use_wandb --num_workers=$num_workers --seed=$seed \
                --dataset=$dataset --batch_size=$batch_size --epochs=$epochs --val_prop=$val_prop \
                --model_name=$model_name --lr=$lr --patience=$patience --sm_where=$sm_where --sm_spectral_norm --sm_transformer  --use_sparse --use_inst_distances \
                > output/${experiment_name}/out_${BASHPID}.txt 2>&1
        done
    done
done

############################################################################################################