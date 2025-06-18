import torch # Not strictly needed in this file anymore, but kept for consistency if other torch utils are added
import argparse
import numpy as np
from tqdm import tqdm
from pathlib import Path
from sklearn.cluster import KMeans
from sklearn.neighbors import kneighbors_graph, radius_neighbors_graph
import scipy.sparse as sp # For sp.eye in KNN graph
import logging

try:
    from processing.utils import dump_json # Assuming this might exist for other purposes
except ImportError:
    import json
    def dump_json(data, filepath):
        """Fallback JSON dumping function."""
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=4)
    logging.info("Using a fallback JSON dumper as processing.utils.dump_json was not found.")

# Logger Setup
def setup_logger():
    """Sets up a basic console logger for the script."""
    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s] %(levelname)s - %(module)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    return logging.getLogger(__name__)

logger = setup_logger()

# Patch-Level Processing Functions
def clustering(feats, num_clusters):
    """
    Apply KMeans clustering to patch feature vectors.
    """
    if feats is None:
        logger.warning("Input features are None, skipping K-Means clustering.")
        return None
    if len(feats) == 0:
        logger.warning("Input features array is empty, skipping K-Means clustering.")
        return None
    if num_clusters <= 0:
        logger.info(f"num_clusters is {num_clusters}, skipping K-Means clustering.")
        return None
    
    actual_num_clusters = num_clusters
    if num_clusters > len(feats):
        logger.warning(f"Number of clusters ({num_clusters}) is > number of features ({len(feats)}). Adjusting to {len(feats)}.")
        actual_num_clusters = len(feats)
    
    if actual_num_clusters == 0: # Should only happen if len(feats) was 0 initially
        logger.warning("Adjusted number of clusters is 0. Skipping K-Means.")
        return None

    try:
        # Use 'lloyd' for algorithm if n_init is explicitly set, common for older sklearn
        # For newer sklearn, n_init='auto' is preferred.
        kmeans = KMeans(n_clusters=actual_num_clusters, random_state=42, n_init='auto' if hasattr(KMeans(), 'n_init') and KMeans().n_init == 'auto' else 10)
        cluster_labels = kmeans.fit_predict(feats)
        return cluster_labels.reshape(-1, 1) # Return as [N, 1]
    except Exception as e:
        logger.error(f"Error during K-Means clustering: {e}", exc_info=True)
        return None

def save_cluster_info_to_json(cluster_indices_2d, num_clusters_expected, filepath):
    """
    Converts [N,1] cluster indices to a list-of-lists format and saves to JSON.
    """
    if cluster_indices_2d is None:
        logger.warning("Cluster indices are None, cannot save to JSON.")
        return []

    cluster_labels_flat = cluster_indices_2d.flatten()
    
    # Determine the actual number of unique clusters present to size the list_of_lists correctly
    max_cluster_id_present = -1
    if len(cluster_labels_flat) > 0:
        max_cluster_id_present = np.max(cluster_labels_flat)
    
    # The number of lists should accommodate all cluster IDs present, up to num_clusters_expected
    # This handles cases where K-Means might produce fewer clusters than requested.
    effective_num_clusters = max(num_clusters_expected, max_cluster_id_present + 1)
    
    cluster_lists = [[] for _ in range(effective_num_clusters)]
    
    for patch_idx, cluster_id in enumerate(cluster_labels_flat):
        if 0 <= cluster_id < effective_num_clusters:
            cluster_lists[cluster_id].append(int(patch_idx)) # Store as int
        else:
            logger.warning(f"Patch {patch_idx} has invalid cluster ID {cluster_id}. Expected range: [0, {effective_num_clusters-1}].")

    if filepath is not None:
        try:
            dump_json(cluster_lists, filepath)
            logger.info(f"Saved patch cluster assignments (list-of-lists) in JSON format: {filepath}")
        except Exception as e:
            logger.error(f"Error saving JSON file {filepath}: {e}", exc_info=True)
    
    return cluster_lists


def build_spatial_graph(patch_coords, radius_ratio=0.1):
    """
    Builds a spatial graph based on patch coordinates using radius connectivity.
    Returns a dense [N, N] float32 adjacency matrix.
    """
    num_patches = len(patch_coords)
    if patch_coords is None or num_patches == 0:
        logger.debug("Patch coordinates are empty or None, cannot build spatial graph.")
        return np.array([], dtype=np.float32).reshape(0,0) if patch_coords is not None else None

    if num_patches == 1:
        return np.array([[1.0]], dtype=np.float32) # Self-loop for a single node

    try:
        coord_range = np.ptp(patch_coords, axis=0)
        max_range = np.max(coord_range) if len(coord_range) > 0 and np.all(np.isfinite(coord_range)) else 0.0
        
        radius = 0.0
        if max_range > 0:
            radius = radius_ratio * max_range
        elif num_patches > 1 : # All points are the same or very close
            logger.warning(f"Max coordinate range is {max_range} for {num_patches} patches. Spatial graph might be fully connected or only self-loops depending on radius_ratio.")
            # If all points are the same, radius_neighbors_graph with radius > 0 connects all.
            # If radius is 0, it only connects self.
            radius = 1.0 if radius_ratio > 0 else 0.0 # Default to 1.0 to connect, or 0 for self-loops only
        
        if radius <= 0 and radius_ratio > 0 and max_range > 0:
            logger.warning(f"Calculated radius is <= 0 (value: {radius}) with radius_ratio {radius_ratio} and max_range {max_range}. Using a small default radius of 1.0.")
            radius = 1.0
        elif radius < 0: # Should not happen if radius_ratio is positive
             radius = 0.0

        logger.debug(f"Building spatial graph for {num_patches} patches with radius: {radius:.2f}")
        adj_mat_sparse = radius_neighbors_graph(patch_coords, radius=radius, mode='connectivity', include_self=True)
        adj_dense = adj_mat_sparse.toarray().astype(np.float32)
        
        assert adj_dense.shape == (num_patches, num_patches), \
            f"Spatial graph shape mismatch: expected ({num_patches}, {num_patches}), got {adj_dense.shape}"
        return adj_dense
    except Exception as e:
        logger.error(f"Error building spatial graph: {e}", exc_info=True)
        return None

def build_knn_graph(features, k=5, add_self_loops=True):
    """
    Builds a K-Nearest Neighbors (KNN) graph based on node features.
    Returns a dense [N, N] float32 adjacency matrix.
    """
    num_nodes = features.shape[0]
    if features is None or num_nodes == 0:
        logger.debug("Features are empty or None, cannot build KNN graph.")
        return np.array([], dtype=np.float32).reshape(0,0) if features is not None else None

    if num_nodes == 1:
        return np.array([[1.0 if add_self_loops else 0.0]], dtype=np.float32)

    actual_k = k
    if num_nodes <= actual_k: # k must be < num_nodes for kneighbors_graph
        logger.debug(f"Number of features ({num_nodes}) is <= k ({actual_k}). Adjusting k to {num_nodes - 1}.")
        actual_k = num_nodes - 1 
    
    if actual_k <= 0: # If k becomes 0 or less (e.g., only 1 node and k was 1, adjusted to 0)
        logger.debug(f"Adjusted k is {actual_k}. Building graph with only self-loops (if enabled) or no edges.")
        adj_mat = np.zeros((num_nodes, num_nodes), dtype=np.float32)
        if add_self_loops:
            np.fill_diagonal(adj_mat, 1.0)
        return adj_mat

    try:
        logger.debug(f"Building KNN graph for {num_nodes} nodes with k={actual_k}")
        # include_self=False initially, as we handle self-loops explicitly for clarity
        adj_mat_sparse = kneighbors_graph(features, n_neighbors=actual_k, mode='connectivity', include_self=False)
        if add_self_loops:
            adj_mat_sparse = adj_mat_sparse + sp.eye(adj_mat_sparse.shape[0], format='csr', dtype=np.float32)
        
        adj_dense = adj_mat_sparse.toarray().astype(np.float32)
        assert adj_dense.shape == (num_nodes, num_nodes), \
            f"KNN graph shape mismatch: expected ({num_nodes}, {num_nodes}), got {adj_dense.shape}"
        return adj_dense
    except Exception as e:
        logger.error(f"Error building KNN graph with k={actual_k}: {e}", exc_info=True)
        return None

def adjacency_to_edge_index(adj_mat, threshold=0.0):
    """
    Convert dense adjacency matrix to PyTorch Geometric edge_index format.
    """
    if adj_mat is None or adj_mat.size == 0:
        logger.debug("Adjacency matrix is None or empty, cannot convert to edge index.")
        return None, None 
    
    if not isinstance(adj_mat, np.ndarray):
        adj_mat = np.array(adj_mat, dtype=np.float32)

    rows, cols = np.where(adj_mat > threshold)
    
    if len(rows) == 0:
        logger.debug("No edges found above threshold in adjacency matrix.")
        return np.array([], dtype=np.int64).reshape(2, 0), np.array([], dtype=np.float32)
    
    edge_index = np.stack([rows, cols], axis=0).astype(np.int64)
    edge_weights = adj_mat[rows, cols].astype(np.float32)
    
    return edge_index, edge_weights

# Main Processing Function
def run(args):
    """Main processing loop for patch clustering and adjacency matrix generation."""
    logger.info(f"Starting patch data processing with arguments: {args}")

    feat_dir = Path(args.feat_dir)
    if not feat_dir.is_dir():
        logger.error(f"Feature directory does not exist or is not a directory: {feat_dir}")
        return

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Define the output directory for consolidated patch-level NPZ files
    # This directory will contain one NPZ per WSI, holding all patch-derived data.
    patch_data_output_dir = save_dir / f'k-means-{args.num_clusters}'
    patch_data_output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Outputting consolidated patch data to: {patch_data_output_dir}")

    img_features_npz_files = sorted(list(feat_dir.glob("*.npz")))
    if not img_features_npz_files:
        logger.error(f"No NPZ files found in the feature directory: {feat_dir}")
        return

    logger.info(f"Found {len(img_features_npz_files)} WSI NPZ files to process.")

    for raw_feat_npz_path in tqdm(img_features_npz_files, desc="Processing WSIs"):
        case_id = raw_feat_npz_path.stem
        logger.info(f"--- Processing WSI: {case_id}")

        # Path for the single, consolidated output NPZ for this WSI's patch data
        consolidated_patch_npz_path = patch_data_output_dir / f'{case_id}.npz'

        if consolidated_patch_npz_path.exists() and not args.exist_ok:
            logger.info(f"Consolidated patch data for {case_id} already exists at {consolidated_patch_npz_path}. Skipping.")
            continue
        
        try:
            with np.load(str(raw_feat_npz_path), allow_pickle=True) as loaded_data:
                img_feats = loaded_data.get('img_features')
                coords = loaded_data.get('coords') # Coords might be None or empty

            if img_feats is None:
                logger.warning(f"Skipping {case_id}: 'img_features' key not found in {raw_feat_npz_path}.")
                continue
            
            num_total_patches = img_feats.shape[0]
            if num_total_patches == 0:
                logger.warning(f"Skipping {case_id}: No patches (img_features is empty).")
                continue

            logger.info(f"Loaded {num_total_patches} patches with {img_feats.shape[1]}-dim features for {case_id}.")
            if coords is not None:
                logger.info(f"Coordinates loaded with shape: {coords.shape}")
            else:
                logger.info("No coordinates found in the input NPZ.")

            # 1. Patch-Level Clustering
            patch_cluster_labels_1d = None # To store 1D array [N_patches]
            if args.num_clusters > 0:
                logger.info(f"Performing K-Means clustering with up to {args.num_clusters} clusters...")
                # clustering() returns [N,1] or None
                features_cluster_indices_2d = clustering(img_feats, args.num_clusters) 
                if features_cluster_indices_2d is not None:
                    patch_cluster_labels_1d = features_cluster_indices_2d.flatten() # Convert to 1D
                    logger.info(f"K-Means clustering completed for {case_id}. Found {len(np.unique(patch_cluster_labels_1d))} unique clusters.")
                    
                    # Optional: Save cluster assignments as JSON (list-of-lists format)
                    # This is mostly for human readability or other tools, not directly used by WSIWithCluster if NPZ is primary
                    if args.save_cluster_json:
                        patch_json_output_path = patch_data_output_dir / f'{case_id}_clusters.json'
                        save_cluster_info_to_json(features_cluster_indices_2d, args.num_clusters, patch_json_output_path)
                else:
                    logger.warning(f"K-Means clustering failed for {case_id}. No patch cluster labels will be saved.")
            else:
                logger.info(f"K-Means clustering skipped for {case_id} (num_clusters <= 0).")


            # 2. Patch-Level Adjacency Matrix Generation
            actual_patch_adj_mat_NxN = None # This will be the [N, N] dense matrix
            if args.adj_mat_type != 'none' and num_total_patches > 0:
                logger.info(f"Attempting to generate patch-level '{args.adj_mat_type}' adjacency matrix for {case_id}...")
                
                if args.adj_mat_type == 'spatial':
                    if coords is not None and coords.shape[0] == num_total_patches:
                        actual_patch_adj_mat_NxN = build_spatial_graph(coords, radius_ratio=args.spatial_radius_ratio)
                    else:
                        logger.warning(f"Cannot generate spatial graph for {case_id}: Coordinates are missing, empty, or mismatched. Coords shape: {coords.shape if coords is not None else 'None'}, Patches: {num_total_patches}")
                elif args.adj_mat_type == 'knn':
                    actual_patch_adj_mat_NxN = build_knn_graph(img_feats, k=args.knn_k)
                
                if actual_patch_adj_mat_NxN is not None:
                    expected_shape = (num_total_patches, num_total_patches)
                    if actual_patch_adj_mat_NxN.shape == expected_shape:
                        logger.info(f"Successfully generated patch-level '{args.adj_mat_type}' adjacency matrix with shape {actual_patch_adj_mat_NxN.shape}.")
                    else:
                        # This case should ideally be caught by assertions within build_..._graph functions
                        logger.error(f"CRITICAL ERROR: Generated patch_adj_mat for {case_id} has UNEXPECTED shape {actual_patch_adj_mat_NxN.shape}, expected {expected_shape}. Setting to None.")
                        actual_patch_adj_mat_NxN = None # Do not save incorrect matrix
                else:
                    logger.warning(f"Failed to generate patch-level '{args.adj_mat_type}' adjacency matrix for {case_id}.")
            elif args.adj_mat_type == 'none':
                logger.info(f"Adjacency matrix generation skipped for {case_id} (adj_mat_type='none').")
            else: # num_total_patches == 0
                 logger.info(f"Skipping adjacency matrix generation for {case_id} as there are no patches.")


            # 3. Prepare Data and Save Consolidated Patch-Level NPZ
            data_to_save = {
                'img_features': img_feats, # [N, D_feat]
                'num_patches': np.array([num_total_patches], dtype=np.int32), # Scalar in array for consistency
            }

            if coords is not None:
                data_to_save['coords'] = coords # [N, 2]
            else: # Save an empty array if coords were None, to maintain key presence
                data_to_save['coords'] = np.array([], dtype=np.float32).reshape(0,2 if num_total_patches > 0 else 0)


            if patch_cluster_labels_1d is not None:
                # This is the 1D array [N_patches] of cluster labels
                data_to_save['patch_clusters'] = patch_cluster_labels_1d 
                assert patch_cluster_labels_1d.ndim == 1 and patch_cluster_labels_1d.shape[0] == num_total_patches, \
                    "Internal check failed: 'patch_clusters' must be 1D and match num_total_patches"

            if actual_patch_adj_mat_NxN is not None:
                # This is the [N, N] dense adjacency matrix
                data_to_save['patch_adj_mat'] = actual_patch_adj_mat_NxN
                
                # Optionally, also save in edge_index format if specified
                if args.save_edge_format:
                    edge_index, edge_weights = adjacency_to_edge_index(actual_patch_adj_mat_NxN)
                    if edge_index is not None: # edge_index can be (2,0) if no edges
                        data_to_save['edge_index'] = edge_index
                        if edge_weights is not None: # edge_weights can be (0,)
                             data_to_save['edge_weights'] = edge_weights
                        logger.info(f"Saved edge_index (shape {edge_index.shape}) and edge_weights (shape {edge_weights.shape if edge_weights is not None else 'None'}) for {case_id}.")


            try:
                np.savez_compressed(consolidated_patch_npz_path, **data_to_save)
                logger.info(f"Successfully saved consolidated patch data for {case_id} to: {consolidated_patch_npz_path}")
                logger.info(f"NPZ contains keys: {list(data_to_save.keys())}")

            except Exception as e:
                logger.error(f"Error saving consolidated NPZ for {case_id} to {consolidated_patch_npz_path}: {e}", exc_info=True)

        except FileNotFoundError:
            logger.error(f"Raw feature NPZ file not found: {raw_feat_npz_path}")
        except Exception as e_outer:
            logger.error(f"An unexpected error occurred while processing {case_id} ({raw_feat_npz_path}): {e_outer}", exc_info=True)
            # Continue to the next file
            
    logger.info(f"Finished processing all WSIs. Consolidated patch data saved in {patch_data_output_dir}.")


def main():
    parser = argparse.ArgumentParser(description="Consolidate WSI patch features, perform K-Means clustering, and generate patch-level adjacency matrices.")
    
    parser.add_argument('--feat_dir', type=str, required=True,
                        help='Directory containing RAW NPZ feature files (e.g., from feature extraction). Each NPZ should have at least \'img_features\' and optionally \'coords\'.')
    parser.add_argument('--save_dir', type=str, required=True,
                        help='Base directory to save the consolidated patch-level NPZ files.')
    parser.add_argument('--num_clusters', type=int, default=10,
                        help='Number of K-Means clusters for patch clustering. Set to 0 to skip clustering.')
    parser.add_argument('--adj_mat_type', type=str, default='spatial', 
                        choices=['none', 'spatial', 'knn'],
                        help="Type of patch-level adjacency matrix to generate. 'none' skips generation.")
    parser.add_argument('--spatial_radius_ratio', type=float, default=0.1,
                        help='Fraction of max coordinate range to use as spatial connectivity radius (for adj_mat_type=\'spatial\').')
    parser.add_argument('--knn_k', type=int, default=10,
                        help='Number of nearest neighbors for KNN graph construction (for adj_mat_type=\'knn\').')
    parser.add_argument('--save_edge_format', action='store_true', default=False,
                        help='If set, also save adjacency matrix in edge_index and edge_weights format within the NPZ.')
    parser.add_argument('--save_cluster_json', action='store_true', default=False,
                        help='If set, also save patch cluster assignments (list-of-lists) as a separate JSON file for easier inspection.')
    parser.add_argument('--exist_ok', action='store_true', default=False,
                        help='If set, overwrite existing output NPZ files. Otherwise, skip if output exists.')

    args = parser.parse_args()

    # Argument validation
    if args.num_clusters < 0:
        logger.error("Number of clusters (num_clusters) must be non-negative (0 to skip).")
        return
    if args.adj_mat_type == 'spatial' and args.spatial_radius_ratio <= 0:
        logger.warning("spatial_radius_ratio is <= 0. For spatial graphs, this might lead to no edges or only self-loops unless all points are identical.")
    if args.adj_mat_type == 'knn' and args.knn_k <= 0:
        logger.error("KNN k (knn_k) must be positive for adj_mat_type='knn'.")
        return

    run(args)

if __name__ == '__main__':
    main()