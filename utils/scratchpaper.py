# import os
# from murclcsv import create_murcl_csv

# # Base directories
# base_dir = "/projects/0/prjs1477/SG-MuRCL/data/CAMELYON16-baseline"

# # Generate training CSV
# train_feature_dir = os.path.join(base_dir, "training/features/resnet18")
# train_cluster_dir = os.path.join(base_dir, "training/features/resnet18/k-means-10")
# train_output_file = os.path.join(base_dir, "training/murcl-input_10.csv")

# create_murcl_csv(train_feature_dir, train_cluster_dir, train_output_file)

# # Generate test CSV
# test_feature_dir = os.path.join(base_dir, "test/features/resnet18")
# test_cluster_dir = os.path.join(base_dir, "test/features/resnet18/k-means-10")
# test_output_file = os.path.join(base_dir, "test/murcl-input_10.csv")

# create_murcl_csv(test_feature_dir, test_cluster_dir, test_output_file)


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
    # Set paths
    train_csv = "/projects/0/prjs1477/SG-MuRCL/data/CAMELYON16-baseline/training/murcl-input_10.csv"
    test_csv = "/projects/0/prjs1477/SG-MuRCL/data/CAMELYON16-baseline/test/murcl-input_10.csv"
    output_csv = "/projects/0/prjs1477/SG-MuRCL/data/CAMELYON16-baseline/murcl-input_combined.csv"
    
    # Combine CSV files
    combine_csv_files(train_csv, test_csv, output_csv)