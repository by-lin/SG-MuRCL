import os
import pandas as pd
import glob
from pathlib import Path
import argparse

def create_murcl_csv(feature_dir, cluster_dir, output_dir, output_name):
    """Create a MuRCL-compatible CSV file from feature and cluster files"""
    # Get all feature files
    feature_files = sorted(glob.glob(os.path.join(feature_dir, "*.npz")))
    
    # Create dataset list
    dataset = []
    for feature_file in feature_files:
        case_id = Path(feature_file).stem
        
        # Determine label (0 for normal, 1 for tumor)
        label = 1 if "tumor" in case_id.lower() else 0
        
        # Paths to cluster files
        cluster_npz = os.path.join(cluster_dir, f"{case_id}.npz")
        cluster_json = os.path.join(cluster_dir, f"{case_id}.json")
        
        # Skip if cluster files don't exist
        if not os.path.exists(cluster_npz) or not os.path.exists(cluster_json):
            print(f"Warning: Cluster files for {case_id} not found. Skipping.")
            continue
        
        # Add to dataset
        dataset.append({
            'case_id': case_id,
            'features_filepath': feature_file,
            'label': label,
            'clusters_filepath': cluster_npz,
            'clusters_json_filepath': cluster_json
        })
    
    # Create and save DataFrame
    df = pd.DataFrame(dataset)
    
    # Create output directory if needed
    os.makedirs(output_dir, exist_ok=True)    
    output_file = os.path.join(output_dir, output_name + '.csv')
    
    # Save CSV
    df.to_csv(output_file, index=False)
    print(f"Created CSV file at {output_file} with {len(df)} entries")
    
    return df

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Create MuRCL input CSV files')
    parser.add_argument('--feature_dir', type=str, required=True, help='Directory containing feature files')
    parser.add_argument('--cluster_dir', type=str, required=True, help='Directory containing cluster files')
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory for the CSV file')
    parser.add_argument('--output_name', type=str, default="input", help='Name for the output CSV file (do not include .csv)')
    args = parser.parse_args()
    
    create_murcl_csv(args.feature_dir, args.cluster_dir, args.output_dir, args.output_name)