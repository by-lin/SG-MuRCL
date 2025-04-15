import os
import pandas as pd
import glob
from pathlib import Path

# Paths
feature_dir = "/projects/0/prjs1477/SG-MuRCL/data/CAMELYON16-baseline/training/features/resnet18"
cluster_dir = "/projects/0/prjs1477/SG-MuRCL/data/CAMELYON16-baseline/training/features/resnet18/k-means-10"
output_file = "/projects/0/prjs1477/SG-MuRCL/data/CAMELYON16-baseline/training/murcl-input.csv"

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
df.to_csv(output_file, index=False)
print(f"Created CSV file at {output_file} with {len(df)} entries")