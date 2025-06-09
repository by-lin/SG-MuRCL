import pandas as pd
import os

def add_cluster_paths(input_csv, output_csv):
    """
    Add patch_cluster_filepath and region_cluster_filepath columns to the CSV
    by inserting 'k-means-10' and 'k-regions-10' directories before the filename
    """
    # Read the input CSV
    df = pd.read_csv(input_csv)
    
    def create_cluster_path(original_path, cluster_type):
        """
        Create cluster path by inserting cluster directory before filename
        
        Args:
            original_path: Original features filepath
            cluster_type: 'k-means-10' or 'k-regions-10'
        
        Returns:
            Modified path with cluster directory
        """
        # Split the path into directory and filename
        directory = os.path.dirname(original_path)
        filename = os.path.basename(original_path)
        
        # Create new path with cluster directory
        cluster_path = os.path.join(directory, cluster_type, filename)
        
        return cluster_path
    
    # Add the new columns
    df['patch_cluster_filepath'] = df['features_filepath'].apply(
        lambda x: create_cluster_path(x, 'k-means-10')
    )
    
    df['region_cluster_filepath'] = df['features_filepath'].apply(
        lambda x: create_cluster_path(x, 'k-regions-10')
    )
    
    # Save to output CSV
    df.to_csv(output_csv, index=False)
    
    return df

# Usage
input_file = '/projects/0/prjs1477/SG-MuRCL/data/C16-SGMuRCL/resnet50_input_10.csv'
output_file = '/projects/0/prjs1477/SG-MuRCL/data/C16-SGMuRCL/resnet50_input_with_clusters_10.csv'

# Process the file
df_with_clusters = add_cluster_paths(input_file, output_file)

# Display first few rows to verify
print("First 5 rows of the updated CSV:")
print(df_with_clusters.head())

# Display some examples to verify the path transformation
print("\nExample path transformations:")
for i in range(3):
    original = df_with_clusters.iloc[i]['features_filepath']
    patch_cluster = df_with_clusters.iloc[i]['patch_cluster_filepath']
    region_cluster = df_with_clusters.iloc[i]['region_cluster_filepath']
    
    print(f"\nOriginal: {original}")
    print(f"Patch cluster: {patch_cluster}")
    print(f"Region cluster: {region_cluster}")