import os
import pandas as pd
import glob
from pathlib import Path
import argparse

def load_reference_labels(reference_csv):
    """
    Load reference.csv and return a dict mapping image filename (with .tif) to label (0=negative, 1=positive).
    """
    df = pd.read_csv(reference_csv)
    label_map = {}
    for _, row in df.iterrows():
        # 0 for negative, 1 for anything else
        label = 0 if row['class'].lower() == 'negative' else 1
        label_map[row['image']] = label
    return label_map

def create_murcl_csv(feature_dir, cluster_dir, output_dir, output_name, reference_csv):
    """Create a MuRCL-compatible CSV file from feature and cluster files, using reference.csv for labels"""
    # Load reference labels
    label_map = load_reference_labels(reference_csv)

    # Get all feature files
    feature_files = sorted(glob.glob(os.path.join(feature_dir, "*.npz")))
    
    # Create dataset list
    dataset = []
    for feature_file in feature_files:
        case_id = Path(feature_file).stem

        # Use reference.csv for label assignment
        image_name = f"{case_id}.tif"
        if image_name not in label_map:
            print(f"Warning: {image_name} not found in reference.csv. Skipping.")
            continue
        label = label_map[image_name]
        
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
    # Ensure .csv is only appended once
    if output_name.lower().endswith('.csv'):
        output_name = output_name[:-4]
    output_file = os.path.join(output_dir, output_name + '.csv')
    
    # Save CSV
    df.to_csv(output_file, index=False)
    print(f"Created CSV file at {output_file} with {len(df)} entries")
    
    return df

if __name__ == "__main__":
    DATASET = "C16-SGMuRCL"
    ENCODER = "resnet18"
    TYPE = "train"
    # Set your parameters here
    feature_dir = f"/projects/0/prjs1477/SG-MuRCL/data/{DATASET}/{TYPE}/features/{ENCODER}/npz_files"
    cluster_dir = f"/projects/0/prjs1477/SG-MuRCL/data/{DATASET}/{TYPE}/features/{ENCODER}/k-means-10"
    output_dir = f"/projects/0/prjs1477/SG-MuRCL/data/{DATASET}/{TYPE}"
    output_name = f"{ENCODER}_{TYPE}"
    reference_csv = f"/projects/0/prjs1477/SG-MuRCL/data/CAMELYON16/evaluation/reference.csv"

    create_murcl_csv(feature_dir, cluster_dir, output_dir, output_name, reference_csv)