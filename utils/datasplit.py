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
    parser = argparse.ArgumentParser(description="Create train/val/test splits for MuRCL")
    parser.add_argument('--train_input', type=str, required=True, help='Path to training CSV file')
    parser.add_argument('--test_input', type=str, required=True, help='Path to test CSV file')
    parser.add_argument('--output_dir', type=str, required=True, help='Path to output JSON file')
    parser.add_argument('--output_name', type=str, default="split", help='Name for the output JSON file (do not include .json)')
    parser.add_argument('--val_ratio', type=float, default=0.2, help='Validation split ratio (default: 0.2)')
    parser.add_argument('--random_seed', type=int, default=985, help='Random seed for reproducibility (default: 985)')
    args = parser.parse_args()

    create_train_val_test_splits(
        args.train_input,
        args.test_input,
        args.output_dir,
        output_name=args.output_name,
        val_ratio=args.val_ratio,
        random_seed=args.random_seed
    )