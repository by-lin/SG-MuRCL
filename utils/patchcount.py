import os
import json
import glob
from pathlib import Path
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np

def analyze_patch_counts(coord_dir, filename, output_dir=None):
    """
    Analyze patch counts for WSIs in the given coord directory
    
    Args:
        coord_dir: Path to directory containing JSON coordinate files
        output_dir: Optional directory to save histogram plot
        filename: Base filename for output histogram plot
    """
    # List all JSON files
    json_files = sorted(glob.glob(os.path.join(coord_dir, "*.json")))
    
    if not json_files:
        print(f"No JSON files found in {coord_dir}")
        return
        
    # Track counts by category (assuming naming pattern for tumor/normal)
    categories = defaultdict(list)
    total_patches = 0
    slide_counts = []
    
    print(f"{'Slide Name':<40} | {'Category':<10} | {'# Patches':<10}")
    print("-" * 65)
    
    # Process each file
    for json_file in json_files:
        with open(json_file, 'r') as f:
            data = json.load(f)
            slide_name = Path(json_file).stem
            num_patches = data['num_patches']
            total_patches += num_patches
            
            # Determine category based on filename
            if "tumor" in slide_name.lower():
                category = "tumor"
            elif "normal" in slide_name.lower():
                category = "normal"
            else:
                category = "unknown"
                
            categories[category].append(num_patches)
            slide_counts.append(num_patches)
            
            print(f"{slide_name:<40} | {category:<10} | {num_patches:<10}")
    
    print("-" * 65)
    print(f"{'TOTAL':<40} | {'all':<10} | {total_patches:<10}")
    
    slide_count = len(json_files)
    if slide_count > 0:
        avg_patches = total_patches / slide_count
        print(f"{'AVERAGE':<40} | {'all':<10} | {avg_patches:<10.1f}")
        
        # Print category statistics
        for category, counts in categories.items():
            if counts:
                avg = sum(counts) / len(counts)
                print(f"{'AVERAGE ' + category.upper():<40} | {category:<10} | {avg:<10.1f}")
    
    # Create histogram if output directory is specified
    if output_dir and slide_counts:
        os.makedirs(output_dir, exist_ok=True)
        plt.figure(figsize=(10, 6))
        plt.hist(slide_counts, bins=20, alpha=0.75, color='steelblue')
        plt.xlabel('Patches per Slide')
        plt.ylabel('Number of Slides')
        plt.title('Distribution of Patch Counts per Slide')
        plt.axvline(np.mean(slide_counts), color='red', linestyle='dashed', 
                   linewidth=1, label=f'Mean: {np.mean(slide_counts):.1f}')
        plt.axvline(np.median(slide_counts), color='green', linestyle='dashed', 
                   linewidth=1, label=f'Median: {np.median(slide_counts):.1f}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, filename + '_patch_distribution.png'))
        print(f"\nHistogram saved to {os.path.join(output_dir, 'patch_distribution.png')}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Analyze patch counts from WSI coordinate files')
    parser.add_argument('--coord_dir', type=str, required=True, 
                        help='Directory containing JSON coordinate files')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Directory to save histogram plot (optional)')
    parser.add_argument('--filename', type=str, default='patch_counts',
                        help='Base filename for output histogram plot')
    
    args = parser.parse_args()
    analyze_patch_counts(args.coord_dir, args.filename, args.output_dir)