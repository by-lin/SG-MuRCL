import os
import json
import numpy as np
import pandas as pd
import argparse

def create_train_val_test_splits(train_input, test_input, output_dir, output_name, val_ratio, random_seed):
    """Create train/validation/test splits for MuRCL"""
    # Load the training CSV file
    train_df = pd.read_csv(train_input)
    
    # Get case IDs and labels for training data
    normal_cases = train_df[train_df['label'] == 0]['case_id'].tolist()
    tumor_cases = train_df[train_df['label'] == 1]['case_id'].tolist()
    
    # Set random seed for reproducibility
    np.random.seed(random_seed)
    
    # Shuffle the cases
    np.random.shuffle(normal_cases)
    np.random.shuffle(tumor_cases)
    
    # Split into train and validation sets
    normal_val_count = int(len(normal_cases) * val_ratio)
    tumor_val_count = int(len(tumor_cases) * val_ratio)
    
    normal_train = normal_cases[:-normal_val_count]
    normal_val = normal_cases[-normal_val_count:]
    
    tumor_train = tumor_cases[:-tumor_val_count]
    tumor_val = tumor_cases[-tumor_val_count:]
    
    # Combine train and val sets
    train_cases = normal_train + tumor_train
    val_cases = normal_val + tumor_val
    
    # Handle test data
    test_df = pd.read_csv(test_input)
    test_cases = test_df['case_id'].tolist()
    
    # Create the splits dictionary
    splits = {
        'train': train_cases,
        'valid': val_cases,
        'test': test_cases
    }
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    # Ensure .json is only appended once
    output_json = os.path.join(output_dir, output_name + '.json')
    
    # Save to JSON file
    with open(output_json, 'w') as f:
        json.dump(splits, f, indent=2)
    
    print(f"Created splits file at {output_json}")
    print(f"Train: {len(train_cases)} slides ({len(normal_train)} normal, {len(tumor_train)} tumor)")
    print(f"Valid: {len(val_cases)} slides ({len(normal_val)} normal, {len(tumor_val)} tumor)")
    print(f"Test: {len(test_cases)} slides")

if __name__ == "__main__":
    # Set your parameters here
    DATASET = "C16-SGMuRCL"
    ENCODER = "resnet18"
    train_input = f"/projects/0/prjs1477/SG-MuRCL/data/{DATASET}/train/{ENCODER}_train.csv"
    test_input = f"/projects/0/prjs1477/SG-MuRCL/data/{DATASET}/test/{ENCODER}_test.csv"
    output_dir = f"/projects/0/prjs1477/SG-MuRCL/data/{DATASET}"
    output_name = f"{ENCODER}_split_10"
    val_ratio = 0.2
    random_seed = 985

    create_train_val_test_splits(
        train_input,
        test_input,
        output_dir,
        output_name=output_name,
        val_ratio=val_ratio,
        random_seed=random_seed
    )