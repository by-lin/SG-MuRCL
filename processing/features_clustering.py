import torch
import argparse
import numpy as np
from tqdm import tqdm
from pathlib import Path
from sklearn.cluster import KMeans
from sklearn.neighbors import kneighbors_graph, radius_neighbors_graph
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components as scipy_connected_components
import logging

try:
    from processing.utils import dump_json
except ImportError:
    import json
    def dump_json(data, filepath):
        """Fallback JSON dumping function if .utils.dump_json is not found."""
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=4)
    logging.warning("Failed to import .utils.dump_json, using a fallback JSON dumper.")

# --- Logger Setup ---
def setup_logger():
    """Sets up a basic console logger for the script."""
    logging.basicConfig(
        level=logging.INFO, # Log messages at INFO level and above (WARNING, ERROR, CRITICAL)
        format='[%(asctime)s] %(levelname)s - %(message)s', # Log message format
        datefmt='%Y-%m-%d %H:%M:%S' # Timestamp format
    )
    return logging.getLogger(__name__)

logger = setup_logger() # Global logger instance

# --- Patch-Level Processing Functions ---
def clustering(feats, num_clusters):
    """
    Apply KMeans clustering to patch feature vectors.

    Args:
        feats (np.ndarray): Input feature array (N_patches x D_features).
        num_clusters (int): The desired number of K-Means clusters.

    Returns:
        np.ndarray or None: An array of cluster assignments (N_patches x 1) for each patch,
                             or None if clustering cannot be performed.
    """
    if feats is None:
        logger.warning("Input features are None, skipping K-Means clustering.")
        return None
    if len(feats) == 0:
        logger.warning("Input features array is empty, skipping K-Means clustering.")
        return None
    if num_clusters <= 0:
        logger.warning(f"Invalid number of clusters: {num_clusters}. Skipping K-Means clustering.")
        return None
    if num_clusters >= len(feats):
        logger.warning(f"Number of clusters ({num_clusters}) is >= number of features ({len(feats)}). Adjusting to {len(feats) - 1}.")
        num_clusters = max(1, len(feats) - 1)

    try:
        kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(feats)
        return cluster_labels.reshape(-1, 1)
    except Exception as e:
        logger.error(f"Error during K-Means clustering: {e}")
        return None

def save_to_json(cluster_indices, num_clusters, filepath):
    """
    Converts cluster indices to a list-of-lists format and saves to JSON.

    Args:
        cluster_indices (np.ndarray): Cluster assignment for each patch (N_patches x 1).
        num_clusters (int): Expected number of clusters.
        filepath (Path or None): Path to save the JSON file. If None, just return the data.

    Returns:
        list: List of lists, where each inner list contains patch indices for that cluster.
    """
    if cluster_indices is None:
        logger.warning("Cluster indices are None, cannot save to JSON.")
        return []

    cluster_labels_flat = cluster_indices.flatten()
    cluster_lists = [[] for _ in range(num_clusters)]
    
    for patch_idx, cluster_id in enumerate(cluster_labels_flat):
        if 0 <= cluster_id < num_clusters:
            cluster_lists[cluster_id].append(patch_idx)
        else:
            logger.warning(f"Patch {patch_idx} has invalid cluster ID {cluster_id}. Expected range: [0, {num_clusters-1}].")

    if filepath is not None:
        try:
            dump_json(cluster_lists, filepath)
            logger.info(f"Saved cluster assignments in JSON format: {filepath}")
        except Exception as e:
            logger.error(f"Error saving JSON file {filepath}: {e}")
    
    return cluster_lists

def build_spatial_graph(patch_coords, radius_ratio=0.1):
    """
    Builds a spatial graph based on patch coordinates using radius connectivity.

    Args:
        patch_coords (np.ndarray): Patch coordinates (N_patches x 2).
        radius_ratio (float): Fraction of the maximum coordinate range to use as radius.

    Returns:
        np.ndarray or None: Dense adjacency matrix (N_patches x N_patches) as float32, or None on failure.
    """
    if patch_coords is None or len(patch_coords) == 0:
        logger.debug("Patch coordinates are empty or None, cannot build spatial graph.")
        return None
    if len(patch_coords) == 1:
        return np.array([[1.0]], dtype=np.float32)

    try:
        # Calculate the radius based on coordinate range
        coord_range = np.ptp(patch_coords, axis=0)  # Peak-to-peak (max - min) for each dimension
        max_range = np.max(coord_range)
        radius = radius_ratio * max_range
        
        if radius <= 0:
            logger.warning("Calculated radius is <= 0, using default radius of 1.0")
            radius = 1.0

        logger.debug(f"Using spatial radius: {radius:.2f}")
        
        adj_mat_sparse = radius_neighbors_graph(patch_coords, radius=radius, mode='connectivity', include_self=True)
        return adj_mat_sparse.toarray().astype(np.float32)
    except Exception as e:
        logger.error(f"Error building spatial graph with radius {radius}: {e}")
        return None

def build_knn_graph(features, k=5, add_self_loops=True):
    """
    Builds a K-Nearest Neighbors (KNN) graph based on node features.

    Args:
        features (np.ndarray): Feature array (N_nodes x D_features).
        k (int): Number of nearest neighbors.
        add_self_loops (bool): Whether to add self-loops.

    Returns:
        np.ndarray or None: Dense adjacency matrix (N_nodes x N_nodes) as float32, or None on failure.
    """
    if features is None or len(features) == 0:
        logger.debug("Features are empty or None, cannot build KNN graph.")
        return None if len(features) > 0 else np.array([], dtype=np.float32).reshape(0,0)
    if len(features) == 1 and add_self_loops: # Single node graph
        return np.array([[1.0]], dtype=np.float32)
    if len(features) == 1 and not add_self_loops:
        return np.array([[0.0]], dtype=np.float32)

    actual_k = k
    if features.shape[0] <= actual_k:
        logger.debug(f"Number of features ({features.shape[0]}) is <= k ({actual_k}). Adjusting k.")
        if features.shape[0] <= 1:
            logger.debug("Cannot build KNN graph for 0 or 1 sample with k > 0.")
            return None # Or handle as per single node case above if k=0 was intended
        actual_k = features.shape[0] - 1 
        if actual_k <= 0:
            logger.debug(f"Adjusted k is {actual_k}, cannot build KNN graph.")
            return None # Fully connected graph with k=N-1, if N>1, is valid. k=0 means no edges.
    
    if actual_k == 0: # No neighbors requested
        adj_mat = np.zeros((features.shape[0], features.shape[0]), dtype=np.float32)
        if add_self_loops:
            np.fill_diagonal(adj_mat, 1.0)
        return adj_mat

    try:
        adj_mat_sparse = kneighbors_graph(features, n_neighbors=actual_k, mode='connectivity', include_self=False)
        if add_self_loops:
            adj_mat_sparse = adj_mat_sparse + sp.eye(adj_mat_sparse.shape[0], format='csr')
        return adj_mat_sparse.toarray().astype(np.float32)
    except Exception as e:
        logger.error(f"Error building KNN graph with k={actual_k}: {e}")
        return None

# NEW: Function to convert adjacency matrix to edge index format for GAT
def adjacency_to_edge_index(adj_mat, threshold=0.0):
    """
    Convert adjacency matrix to PyTorch Geometric edge_index format for GAT.
    
    Args:
        adj_mat (np.ndarray): Dense adjacency matrix (N x N).
        threshold (float): Minimum value to consider as an edge.
    
    Returns:
        tuple: (edge_index, edge_weights) where edge_index is (2, num_edges) and edge_weights is (num_edges,)
    """
    if adj_mat is None or adj_mat.size == 0:
        return None, None
    
    # Find edges (non-zero entries above threshold)
    rows, cols = np.where(adj_mat > threshold)
    
    if len(rows) == 0:
        # No edges found
        return np.array([], dtype=np.int64).reshape(2, 0), np.array([], dtype=np.float32)
    
    edge_index = np.stack([rows, cols], axis=0)  # Shape: (2, num_edges)
    edge_weights = adj_mat[rows, cols]
    
    return edge_index.astype(np.int64), edge_weights.astype(np.float32)

# --- Region-Level Processing Functions ---
def segment_patches_to_regions_graph_based(patch_coords, patch_cluster_labels, num_total_patches, num_patch_clusters, connectivity_radius):
    """
    Segments patches into regions using graph-based Connected Component Analysis (CCA).
    Regions are spatially connected components of patches belonging to the *same* K-Means cluster.

    Args:
        patch_coords (np.ndarray): Coordinates of all patches (N_patches x D_coord).
        patch_cluster_labels (np.ndarray): K-Means cluster label for each patch.
        num_total_patches (int): Total number of patches.
        num_patch_clusters (int): Number of K-Means clusters.
        connectivity_radius (float): Radius for defining spatial connectivity *within* a K-Means cluster.

    Returns:
        tuple: (patch_to_region_map, num_regions, all_region_patch_indices)
            - patch_to_region_map (np.ndarray): Maps each patch index to a global region ID.
            - num_regions (int): Total number of unique regions found.
            - all_region_patch_indices (List[List[int]]): List of patch indices for each region.
    """
    if patch_coords is None or patch_cluster_labels is None:
        logger.warning("Patch coordinates or cluster labels are None, cannot perform region segmentation.")
        return None, 0, []

    patch_to_region_map = np.full(num_total_patches, -1, dtype=int)
    all_region_patch_indices = []
    current_region_id = 0

    patch_cluster_labels_flat = patch_cluster_labels.flatten()

    for cluster_id in range(num_patch_clusters):
        # Get patches belonging to this cluster
        cluster_patch_indices = np.where(patch_cluster_labels_flat == cluster_id)[0]
        
        if len(cluster_patch_indices) == 0:
            logger.debug(f"Cluster {cluster_id} has no patches, skipping.")
            continue
        
        if len(cluster_patch_indices) == 1:
            # Single patch forms its own region
            patch_idx = cluster_patch_indices[0]
            patch_to_region_map[patch_idx] = current_region_id
            all_region_patch_indices.append([patch_idx])
            current_region_id += 1
            continue

        # Extract coordinates for patches in this cluster
        cluster_coords = patch_coords[cluster_patch_indices]
        
        try:
            # Build spatial connectivity graph within this cluster
            adj_mat_sparse = radius_neighbors_graph(cluster_coords, radius=connectivity_radius, mode='connectivity', include_self=False)
            
            # Find connected components
            num_components, component_labels = scipy_connected_components(adj_mat_sparse, directed=False)
            
            # Create regions for each connected component
            for component_id in range(num_components):
                component_patch_indices_local = np.where(component_labels == component_id)[0]
                component_patch_indices_global = cluster_patch_indices[component_patch_indices_local]
                
                # Assign global region ID to these patches
                patch_to_region_map[component_patch_indices_global] = current_region_id
                all_region_patch_indices.append(component_patch_indices_global.tolist())
                current_region_id += 1

        except Exception as e:
            logger.error(f"Error during connected components analysis for cluster {cluster_id}: {e}")
            # Fallback: treat each patch as its own region
            for patch_idx in cluster_patch_indices:
                patch_to_region_map[patch_idx] = current_region_id
                all_region_patch_indices.append([patch_idx])
                current_region_id += 1

    num_regions = current_region_id
    logger.info(f"Segmented {num_total_patches} patches into {num_regions} regions using {num_patch_clusters} K-Means clusters.")
    
    return patch_to_region_map, num_regions, all_region_patch_indices

def compute_region_features_and_coords(img_feats, patch_coords, all_region_patch_indices):
    """
    Computes representative features and coordinates for each region by averaging over constituent patches.

    Args:
        img_feats (np.ndarray): Patch features (N_patches x D_features).
        patch_coords (np.ndarray): Patch coordinates (N_patches x D_coord).
        all_region_patch_indices (List[List[int]]): List of patch indices for each region.

    Returns:
        tuple: (region_features, region_centroids)
            - region_features (np.ndarray): Representative features for each region (N_regions x D_features).
            - region_centroids (np.ndarray): Centroid coordinates for each region (N_regions x D_coord).
    """
    if not all_region_patch_indices:
        logger.warning("No region patch indices provided, cannot compute region features.")
        return None, None

    region_features = []
    region_centroids = []

    for region_patch_indices in all_region_patch_indices:
        if not region_patch_indices:
            logger.warning("Found empty region, skipping.")
            continue
        
        try:
            # Average features over patches in this region
            region_patch_features = img_feats[region_patch_indices]
            region_feature = np.mean(region_patch_features, axis=0)
            region_features.append(region_feature)
            
            # Average coordinates to get centroid
            if patch_coords is not None:
                region_patch_coords = patch_coords[region_patch_indices]
                region_centroid = np.mean(region_patch_coords, axis=0)
                region_centroids.append(region_centroid)
            else:
                region_centroids.append(np.zeros(2))  # Default to origin if no coordinates
                
        except Exception as e:
            logger.error(f"Error computing features for region with indices {region_patch_indices}: {e}")
            continue

    if region_features:
        region_features = np.array(region_features)
        region_centroids = np.array(region_centroids)
    else:
        region_features = None
        region_centroids = None

    return region_features, region_centroids

# --- Main Processing Function ---
def run(args):
    """Main processing loop that performs clustering and adjacency matrix generation."""
    logger.info(f"Starting feature clustering with arguments: {args}")

    # Setup paths
    feat_dir = Path(args.feat_dir)
    if not feat_dir.exists():
        logger.error(f"Feature directory does not exist: {feat_dir}")
        return

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Create output directories
    patch_derived_data_output_dir = save_dir / f'k-means-{args.num_clusters}'
    patch_derived_data_output_dir.mkdir(parents=True, exist_ok=True)

    if args.process_regions:
        region_derived_data_output_dir = save_dir / f'k-regions-{args.num_clusters}'
        region_derived_data_output_dir.mkdir(parents=True, exist_ok=True)

    # Find all NPZ feature files
    img_features_npz_files = sorted(list(feat_dir.glob("*.npz")))
    if not img_features_npz_files:
        logger.error(f"No NPZ files found in {feat_dir}")
        return

    logger.info(f"Found {len(img_features_npz_files)} NPZ files to process.")

    # Process each WSI
    for feat_npz_path in tqdm(img_features_npz_files, desc="Processing WSI files"):
        case_id = feat_npz_path.stem
        logger.info(f"--- Processing WSI: {case_id} ---")

        # Skip if output already exists and not overwriting
        output_patch_npz_path = patch_derived_data_output_dir / f'{case_id}.npz'
        if output_patch_npz_path.exists() and not args.exist_ok:
            logger.info(f"Patch-level output for {case_id} already exists. Skipping...")
            continue

        try:
            # Load feature data
            loaded_data_obj = np.load(str(feat_npz_path), allow_pickle=True)
            original_data_dict = dict(loaded_data_obj)
            loaded_data_obj.close()

            # Extract features and coordinates
            img_feats = original_data_dict.get('img_features')
            coords = original_data_dict.get('coords')
            
            if img_feats is None:
                logger.warning(f"Skipping {case_id}: 'img_features' not found in NPZ file.")
                continue
                
            num_total_patches = img_feats.shape[0]
            if num_total_patches == 0:
                logger.warning(f"Skipping {case_id}: No patches found.")
                continue

            logger.info(f"Loaded {num_total_patches} patches with {img_feats.shape[1]}-dimensional features.")

            # --- 1. Patch-Level Clustering ---
            features_cluster_indices = None
            if args.num_clusters > 0:
                logger.info(f"Performing K-Means clustering with {args.num_clusters} clusters...")
                features_cluster_indices = clustering(img_feats, args.num_clusters)
                if features_cluster_indices is not None:
                    logger.info(f"Successfully completed K-Means clustering.")
                else:
                    logger.warning(f"K-Means clustering failed for {case_id}.")

            # --- 2. Adjacency Matrix Generation ---
            patch_adj_mat = None
            if args.adj_mat_type != 'none':
                logger.info(f"Generating {args.adj_mat_type} adjacency matrix...")
                
                if args.adj_mat_type == 'spatial' and coords is not None:
                    patch_adj_mat = build_spatial_graph(coords, radius_ratio=args.spatial_radius_ratio)
                elif args.adj_mat_type == 'knn':
                    patch_adj_mat = build_knn_graph(img_feats, k=args.knn_k)
                elif args.adj_mat_type == 'spatial' and coords is None:
                    logger.warning(f"Spatial adjacency requested but no coordinates found for {case_id}.")
                
                if patch_adj_mat is not None:
                    logger.info(f"Successfully generated {args.adj_mat_type} adjacency matrix with shape {patch_adj_mat.shape}.")
                else:
                    logger.warning(f"Failed to generate {args.adj_mat_type} adjacency matrix for {case_id}.")

            # --- 3. Enhanced Save Patch-Level Derived Data ---
            data_to_save_in_patch_npz = {
                # Core data for model training
                'img_features': img_feats,
                'coords': coords if coords is not None else np.array([]),
                'num_patches': np.array([num_total_patches]),
            }

            # Clustering data
            if features_cluster_indices is not None:
                data_to_save_in_patch_npz['features_cluster_indices'] = features_cluster_indices
                data_to_save_in_patch_npz['patch_clusters'] = features_cluster_indices.flatten()

            # Adjacency data with edge format for GAT
            if patch_adj_mat is not None:
                data_to_save_in_patch_npz['patch_adj_mat'] = patch_adj_mat
                
                # NEW: Add edge format for GAT
                edge_index, edge_weights = adjacency_to_edge_index(patch_adj_mat)
                if edge_index is not None and edge_index.size > 0:
                    data_to_save_in_patch_npz['edge_index'] = edge_index
                    data_to_save_in_patch_npz['edge_weights'] = edge_weights
                    logger.info(f"Generated {edge_index.shape[1]} edges from adjacency matrix for GAT.")

            # Save enhanced patch-level data
            if data_to_save_in_patch_npz:
                try:
                    np.savez_compressed(output_patch_npz_path, **data_to_save_in_patch_npz)
                    logger.info(f"Saved enhanced patch-level NPZ for {case_id} to {output_patch_npz_path}")
                except Exception as e:
                    logger.error(f"Error saving patch-level NPZ: {e}")

            # Save JSON for backward compatibility
            if features_cluster_indices is not None:
                patch_json_output_path = patch_derived_data_output_dir / f'{case_id}.json'
                save_to_json(features_cluster_indices, args.num_clusters, patch_json_output_path)

            # --- 4. Enhanced Region Processing ---
            if args.process_regions and features_cluster_indices is not None and coords is not None:
                logger.info("Processing regions...")
                
                # Check if region output already exists
                output_region_npz_path = region_derived_data_output_dir / f'{case_id}_regions.npz'
                if output_region_npz_path.exists() and not args.exist_ok:
                    logger.info(f"Region-level output for {case_id} already exists. Skipping region processing...")
                    continue

                # Segment patches into regions
                patch_to_region_map, num_regions, all_region_patch_indices = segment_patches_to_regions_graph_based(
                    coords, features_cluster_indices, num_total_patches, args.num_clusters, 
                    connectivity_radius=args.spatial_radius_ratio * np.max(np.ptp(coords, axis=0))
                )

                if num_regions > 0 and all_region_patch_indices:
                    # Compute region features and coordinates
                    region_features, region_centroids = compute_region_features_and_coords(
                        img_feats, coords, all_region_patch_indices
                    )

                    # Build region-level adjacency matrix
                    region_adj_mat = None
                    if region_centroids is not None and num_regions > 1:
                        if args.adj_mat_type == 'spatial':
                            region_adj_mat = build_spatial_graph(region_centroids, radius_ratio=args.spatial_radius_ratio)
                        elif args.adj_mat_type == 'knn' and region_features is not None:
                            region_adj_mat = build_knn_graph(region_features, k=min(args.knn_k, num_regions-1))

                    # Enhanced region save
                    data_to_save_in_region_npz = {
                        # Core region data
                        'region_features': region_features if region_features is not None else np.array([]),
                        'region_coords': region_centroids if region_centroids is not None else np.array([]),
                        'patch_to_region_map': patch_to_region_map,
                        'num_regions': np.array(num_regions),
                        
                        # Include patch context for convenience
                        'img_features': img_feats,
                        'coords': coords,
                        'patch_clusters': features_cluster_indices.flatten(),
                    }
                    
                    if region_adj_mat is not None:
                        data_to_save_in_region_npz['region_adj_mat'] = region_adj_mat
                        
                        # NEW: Add edge format for region-level GAT
                        region_edge_index, region_edge_weights = adjacency_to_edge_index(region_adj_mat)
                        if region_edge_index is not None and region_edge_index.size > 0:
                            data_to_save_in_region_npz['region_edge_index'] = region_edge_index
                            data_to_save_in_region_npz['region_edge_weights'] = region_edge_weights
                            logger.info(f"Generated {region_edge_index.shape[1]} region edges for GAT.")

                    # Save region data
                    try:
                        np.savez_compressed(output_region_npz_path, **data_to_save_in_region_npz)
                        logger.info(f"Saved enhanced region-level NPZ for {case_id} with {num_regions} regions.")
                    except Exception as e:
                        logger.error(f"Error saving region-level NPZ: {e}")

                else:
                    logger.warning(f"No valid regions found for {case_id}.")

        except Exception as e:
            logger.error(f"Error processing {case_id}: {e}", exc_info=True)
            continue

    logger.info("Feature clustering process completed.")

def main():
    """Parse arguments and run the clustering process."""
    parser = argparse.ArgumentParser(description="Perform K-Means clustering and adjacency matrix generation on WSI patch features for SG-MuRCL.")
    
    parser.add_argument('--feat_dir', type=str, required=True,
                        help='Directory containing NPZ feature files with img_features and coords.')
    parser.add_argument('--save_dir', type=str, required=True,
                        help='Directory to save clustering results and adjacency matrices.')
    parser.add_argument('--num_clusters', type=int, default=10,
                        help='Number of K-Means clusters.')
    parser.add_argument('--adj_mat_type', type=str, default='spatial', 
                        choices=['none', 'spatial', 'knn'],
                        help="Type of patch-level adjacency matrix to generate.")
    parser.add_argument('--spatial_radius_ratio', type=float, default=0.1,
                        help='Fraction of coordinate range to use as spatial connectivity radius.')
    parser.add_argument('--knn_k', type=int, default=10,
                        help='Number of nearest neighbors for KNN graph construction.')
    parser.add_argument('--process_regions', action='store_true', default=True,
                        help='Whether to perform region-level processing via connected components.')
    parser.add_argument('--exist_ok', action='store_true', default=False,
                        help='If set, overwrite existing output files.')

    args = parser.parse_args()

    # Validate arguments
    if args.num_clusters <= 0:
        logger.error("Number of clusters must be positive.")
        return
    if args.spatial_radius_ratio <= 0:
        logger.error("Spatial radius ratio must be positive.")
        return
    if args.knn_k <= 0:
        logger.error("KNN k must be positive.")
        return

    run(args)

if __name__ == '__main__':
    main()