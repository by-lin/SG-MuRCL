import os
import json
import numpy as np
import pandas as pd

def create_train_val_test_splits(train_csv, test_csv, output_json, val_ratio=0.2, random_seed=42):
    """Create train/validation/test splits for MuRCL"""
    # Load the training CSV file
    train_df = pd.read_csv(train_csv)
    
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
    test_df = pd.read_csv(test_csv)
    test_cases = test_df['case_id'].tolist()
    
    # Create the splits dictionary
    splits = {
        'train': train_cases,
        'valid': val_cases,
        'test': test_cases
    }
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_json), exist_ok=True)
    
    # Save to JSON file
    with open(output_json, 'w') as f:
        json.dump(splits, f, indent=2)
    
    print(f"Created splits file at {output_json}")
    print(f"Train: {len(train_cases)} slides ({len(normal_train)} normal, {len(tumor_train)} tumor)")
    print(f"Valid: {len(val_cases)} slides ({len(normal_val)} normal, {len(tumor_val)} tumor)")
    print(f"Test: {len(test_cases)} slides")

if __name__ == "__main__":
    # Input and output paths
    train_csv = "/projects/0/prjs1477/SG-MuRCL/data/CAMELYON16-baseline/training/murcl-input_10.csv"
    test_csv = "/projects/0/prjs1477/SG-MuRCL/data/CAMELYON16-baseline/test/murcl-input_10.csv"
    
    output_dir = "/projects/0/prjs1477/SG-MuRCL/data/CAMELYON16-baseline"
    os.makedirs(output_dir, exist_ok=True)
    output_json = os.path.join(output_dir, "data_splits.json")
    
    # Create the splits
    create_train_val_test_splits(train_csv, test_csv, output_json)