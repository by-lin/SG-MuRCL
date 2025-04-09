import os
from pathlib import Path

# Change this path to your actual h5 directory
h5_dir = Path("/projects/0/prjs1477/SG-MuRCL/data/CAMELYON16-features/h5_files/training")
output_csv = Path("/projects/0/prjs1477/SG-MuRCL/data/h5_training.csv")

# Find all .h5 files, excluding ones with "test" in the filename
h5_files = sorted([
    str(p.resolve()) for p in h5_dir.glob("*.h5")
    if "test" not in p.name.lower()
])

# Save to CSV
with open(output_csv, "w") as f:
    for path in h5_files:
        f.write(f"{path}\n")

print(f"✅ Saved {len(h5_files)} entries to {output_csv}")
