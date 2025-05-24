import pandas as pd
import os
from pathlib import Path

def combine_csv_files(train_csv, test_csv, output_csv):
    """
    Combine training and test CSV files into a single file for MuRCL training
    """
    print(f"Loading training CSV from: {train_csv}")
    train_df = pd.read_csv(train_csv)
    print(f"Found {len(train_df)} entries in training CSV")
    
    print(f"Loading test CSV from: {test_csv}")
    test_df = pd.read_csv(test_csv)
    print(f"Found {len(test_df)} entries in test CSV")
    
    # Combine dataframes
    combined_df = pd.concat([train_df, test_df], ignore_index=True)
    
    # Check for duplicates
    duplicates = combined_df['case_id'].duplicated().sum()
    if duplicates > 0:
        print(f"WARNING: Found {duplicates} duplicate case_ids!")
        # Drop duplicates if needed
        # combined_df = combined_df.drop_duplicates(subset=['case_id'], keep='first')
    else:
        print("No duplicate case_ids found.")
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    
    # Save combined CSV
    combined_df.to_csv(output_csv, index=False)
    print(f"Combined CSV saved to: {output_csv}")
    print(f"Total entries: {len(combined_df)}")

if __name__ == "__main__":
    DATASET = "C16-SGMuRCL"
    ENCODER = "c16x20-simclr-resnet18"
    # Set paths
    train_csv = f"/projects/0/prjs1477/SG-MuRCL/data/{DATASET}/train/{ENCODER}_train.csv"
    test_csv = f"/projects/0/prjs1477/SG-MuRCL/data/{DATASET}/test/{ENCODER}_test.csv"
    output_csv = f"/projects/0/prjs1477/SG-MuRCL/data/{DATASET}/{ENCODER}_input_10.csv"
    
    # Combine CSV files
    combine_csv_files(train_csv, test_csv, output_csv)