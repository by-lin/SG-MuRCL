import pandas as pd
from pathlib import Path

TYPE = "test"  # Change to "test" if needed
ENCODER = "resnet50"  # Change as needed

# Set these paths for train or test as needed
h5_dir = Path(f"/projects/0/prjs1477/SG-MuRCL/data/C16-SGMuRCL/{TYPE}/features/{ENCODER}/h5_files")
reference_csv = Path(f"/projects/0/prjs1477/SG-MuRCL/data/CAMELYON16/evaluation/reference.csv")
output_csv = Path(f"/projects/0/prjs1477/SG-MuRCL/data/C16-SGMuRCL/{TYPE}/features/{ENCODER}/h5{TYPE}.csv")

# Load reference labels
ref_df = pd.read_csv(reference_csv)
label_map = {}
for _, row in ref_df.iterrows():
    label = 0 if row['class'].lower() == 'negative' else 1
    label_map[row['image'].replace('.tif', '')] = label

# Gather h5 files and assign labels
rows = []
for p in sorted(h5_dir.glob("*.h5")):
    case_id = p.stem
    label = label_map.get(case_id, None)
    if label is not None:
        rows.append({'filepath': str(p.resolve()), 'label': label})
    else:
        print(f"Warning: No label found for {case_id}")

# Save to CSV
df = pd.DataFrame(rows)
df.to_csv(output_csv, index=False)
print(f"✅ Saved {len(df)} entries to {output_csv}")