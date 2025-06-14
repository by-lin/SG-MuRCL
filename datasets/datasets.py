import random
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, Iterable, Dict, Union, List, Optional

import torch
from torch.utils.data.dataset import Dataset
from tqdm import tqdm
from utils.general import extract_sub_adjacency_matrix 

# For deriving region cluster labels
from scipy.stats import mode as scipy_mode 
import logging

# Setup logger for this module
logger = logging.getLogger(__name__)
if not logger.hasHandlers():
    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s] %(module)s %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

# Constants
DEFAULT_FEATURE_DIM = 1024
SUPPORTED_GRAPH_LEVELS = ['patch', 'region']

def translate_ppo_action_to_patch_indices(ppo_cluster_action, patch_clusters_indices, num_total_patches_to_select, device):
    """
    Translates a PPO action over clusters into a list of selected patch indices.

    Args:
        ppo_cluster_action (torch.Tensor): Tensor of shape (num_clusters_in_action_space), 
                                           representing scores/probabilities for each cluster.
        patch_clusters_indices (List[List[int]]): List where patch_clusters_indices[c] is a list of 
                                                  patch indices in actual cluster c for the WSI. 
                                                  Length is num_actual_clusters_for_wsi.
        num_total_patches_to_select (int): The total number of patches to select (e.g., args.feat_size).
        device (torch.device): Device to put tensors on.

    Returns:
        torch.Tensor: Tensor of selected patch indices, shape (num_total_patches_to_select,).
                      Returns tensor of -1 if no patches can be selected.
    """
    num_actual_clusters_for_wsi = len(patch_clusters_indices)
    selected_patch_indices_list = []

    if num_actual_clusters_for_wsi == 0 or sum(len(c) for c in patch_clusters_indices) == 0 or num_total_patches_to_select == 0:
        return torch.full((num_total_patches_to_select,), -1, dtype=torch.long, device=device)

    # ppo_cluster_action is for the fixed action space size (e.g., args.num_clusters)
    # We only care about the actions for the actual clusters present in this WSI.
    current_ppo_action_for_wsi = ppo_cluster_action[:num_actual_clusters_for_wsi]
    
    # Filter out empty clusters from patch_clusters_indices and corresponding actions
    valid_cluster_indices_in_wsi = [
        i for i, patches in enumerate(patch_clusters_indices) if patches
    ]

    if not valid_cluster_indices_in_wsi:
        return torch.full((num_total_patches_to_select,), -1, dtype=torch.long, device=device)

    # Get actions and patch lists for only the valid (non-empty) clusters
    action_for_valid_clusters = current_ppo_action_for_wsi[valid_cluster_indices_in_wsi]
    filtered_patch_cluster_lists = [patch_clusters_indices[i] for i in valid_cluster_indices_in_wsi]

    if action_for_valid_clusters.numel() == 0:
        return torch.full((num_total_patches_to_select,), -1, dtype=torch.long, device=device)

    cluster_probs = torch.softmax(action_for_valid_clusters, dim=0)
    
    if not torch.isfinite(cluster_probs).all() or cluster_probs.sum() == 0:
        if len(filtered_patch_cluster_lists) > 0:
            cluster_probs = torch.ones(len(filtered_patch_cluster_lists), device=device) / len(filtered_patch_cluster_lists)
        else:
            return torch.full((num_total_patches_to_select,), -1, dtype=torch.long, device=device)

    try:
        sampled_cluster_draws_indices_in_filtered = torch.multinomial(
            cluster_probs, num_total_patches_to_select, replacement=True
        )
    except RuntimeError:
        if len(filtered_patch_cluster_lists) > 0:
            uniform_probs = torch.ones(len(filtered_patch_cluster_lists), device=device) / len(filtered_patch_cluster_lists)
            sampled_cluster_draws_indices_in_filtered = torch.multinomial(uniform_probs, num_total_patches_to_select, replacement=True)
        else:
            return torch.full((num_total_patches_to_select,), -1, dtype=torch.long, device=device)

    for drawn_idx_in_filtered_list in sampled_cluster_draws_indices_in_filtered:
        actual_cluster_patches = filtered_patch_cluster_lists[drawn_idx_in_filtered_list]
        if actual_cluster_patches:
            selected_patch_original_idx = random.choice(actual_cluster_patches)
            selected_patch_indices_list.append(selected_patch_original_idx)
        else:
            selected_patch_indices_list.append(-1) 

    if not selected_patch_indices_list:
        return torch.empty(0, dtype=torch.long, device=device)
        
    return torch.tensor(selected_patch_indices_list, dtype=torch.long, device=device)


class WSIDataset(Dataset):
    """Basic WSI Dataset, which can obtain the features of each patch of WSIs."""

    def __init__(self,
                 data_csv: Union[str, Path],
                 indices: Optional[Iterable[str]] = None,
                 num_sample_patches: Optional[int] = None,
                 shuffle: bool = False,
                 patch_random: bool = False,
                 preload: bool = True,
                 fixed_size: bool = False,
                 ) -> None:
        super(WSIDataset, self).__init__()
        self.data_csv_path = Path(data_csv)
        self.indices = list(indices) if indices is not None else None
        self.num_sample_patches = num_sample_patches
        self.fixed_size = fixed_size
        self.preload = preload
        self.patch_random = patch_random

        self.samples_df = self._process_data_csv()
        if self.indices is None:
            self.indices = self.samples_df.index.tolist()
        
        if not self.indices:
            logger.warning(f"WSIDataset initialized with no case_ids to process from {data_csv}.")
            self.patch_dim = DEFAULT_FEATURE_DIM
            self.patch_features_cache = {}
            return

        if shuffle:
            self.shuffle()

        # Determine patch_dim
        self.patch_dim = self._determine_feature_dimension()

        # Memory usage warning
        if self.preload and len(self.indices) > 1000:
            estimated_memory = len(self.indices) * self.patch_dim * 4 / 1024**3  # GB estimate
            logger.warning(f"Preloading {len(self.indices)} samples may use significant memory (~{estimated_memory:.2f} GB)")

        self.patch_features_cache: Dict[str, np.ndarray] = {}
        if self.preload:
            self.patch_features_cache = self._preload_patch_features()

    def __len__(self) -> int:
        return len(self.indices) if self.indices else 0

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, str]:
        if not self.indices or index >= len(self.indices):
            raise IndexError(f"Invalid index {index} for dataset size {len(self.indices)}")
            
        case_id = self.indices[index]

        patch_feature_np: Optional[np.ndarray] = None
        if self.preload:
            patch_feature_np = self.patch_features_cache.get(case_id)
        else:
            patch_feature_np = self._load_features(case_id)
        
        if patch_feature_np is None or patch_feature_np.size == 0:
            logger.warning(f"Empty features for {case_id}, returning zero tensor")
            patch_feature_np = np.zeros((0, self.patch_dim), dtype=np.float32)

        patch_feature_np = self._sample_instances(patch_feature_np)
        if self.fixed_size:
            patch_feature_np = self._apply_fixed_size_padding(patch_feature_np)

        patch_feature_tensor = torch.as_tensor(patch_feature_np, dtype=torch.float32)
        label = torch.tensor(self.samples_df.at[case_id, 'label'], dtype=torch.long)
        return patch_feature_tensor, label, case_id

    def shuffle(self) -> None:
        """Shuffle the dataset indices."""
        random.shuffle(self.indices)
        logger.debug(f"Dataset shuffled, total samples: {len(self.indices)}")

    def _process_data_csv(self) -> pd.DataFrame:
        """Process the CSV file and validate required columns."""
        try:
            data_df = pd.read_csv(self.data_csv_path)
            required_cols = ['case_id', 'features_filepath', 'label']
            missing_cols = [col for col in required_cols if col not in data_df.columns]
            if missing_cols:
                raise ValueError(f"Required columns missing from CSV: {missing_cols}")
                
            data_df.set_index(keys='case_id', inplace=True, drop=False)
            
            if self.indices is not None:
                valid_indices = [idx for idx in self.indices if idx in data_df.index]
                if len(valid_indices) != len(self.indices):
                    logger.warning("Some provided indices were not found in the CSV.")
                if not valid_indices:
                    logger.warning("No valid indices provided or found in CSV. Dataset will be empty.")
                    return pd.DataFrame(columns=data_df.columns)
                return data_df.loc[valid_indices].copy()
            return data_df.copy()
        except FileNotFoundError:
            logger.error(f"Data CSV file not found: {self.data_csv_path}")
            raise
        except Exception as e:
            logger.error(f"Error processing data CSV {self.data_csv_path}: {e}")
            raise

    def _determine_feature_dimension(self) -> int:
        """Determine feature dimension from the first valid case."""
        for case_id in self.indices:
            if case_id in self.samples_df.index:
                try:
                    features_filepath = self.samples_df.at[case_id, 'features_filepath']
                    with np.load(features_filepath) as data:
                        return data['img_features'].shape[-1]
                except Exception as e:
                    logger.warning(f"Error loading features for dimension detection from {case_id}: {e}")
                    continue
        
        logger.warning("Could not determine feature dimension, using default 1024")
        return DEFAULT_FEATURE_DIM

    def _load_features(self, case_id: str) -> Optional[np.ndarray]:
        """Load features for a specific case."""
        try:
            features_filepath = self.samples_df.at[case_id, 'features_filepath']
            with np.load(features_filepath) as data:
                return data['img_features']
        except Exception as e:
            logger.error(f"Error loading features for {case_id}: {e}")
            return None
            
    def _preload_patch_features(self) -> Dict[str, np.ndarray]:
        """Preload all patch features into memory."""
        loaded_features: Dict[str, np.ndarray] = {}
        if not self.indices: 
            return loaded_features

        logger.info(f"Preloading patch features for {len(self.indices)} WSIs...")
        for case_id in tqdm(self.indices, desc="Preloading patch features"):
            features = self._load_features(case_id)
            if features is not None:
                loaded_features[case_id] = features
            else:
                loaded_features[case_id] = np.zeros((0, self.patch_dim), dtype=np.float32)
        return loaded_features

    def _sample_instances(self, features_np: np.ndarray) -> np.ndarray:
        """Sample instances from features array."""
        num_patches = features_np.shape[0]
        if self.num_sample_patches is not None and num_patches > self.num_sample_patches:
            sample_indices = np.random.choice(num_patches, size=self.num_sample_patches, replace=False)
            features_np = features_np[sample_indices]
        
        if self.patch_random and features_np.shape[0] > 0:
            np.random.shuffle(features_np)
        return features_np

    def _apply_fixed_size_padding(self, features_np: np.ndarray) -> np.ndarray:
        """Apply fixed size padding or truncation."""
        if self.num_sample_patches is None:
            return features_np

        num_current_patches = features_np.shape[0]
        feat_dim = features_np.shape[1] if num_current_patches > 0 else self.patch_dim

        if feat_dim == 0:
            logger.warning("Patch dimension is 0, cannot apply fixed size padding correctly.")
            return features_np

        if num_current_patches < self.num_sample_patches:
            margin = self.num_sample_patches - num_current_patches
            feat_pad = np.zeros(shape=(margin, feat_dim), dtype=features_np.dtype)
            features_np = np.concatenate((features_np, feat_pad), axis=0)
        elif num_current_patches > self.num_sample_patches:
            features_np = features_np[:self.num_sample_patches]
        return features_np


class WSIWithCluster(WSIDataset):
    """WSI Dataset with clustering information for MuRCL training."""
    
    def __init__(self, 
                data_csv: Union[str, Path], 
                indices: Optional[Iterable[str]] = None,
                graph_level: str = 'patch',
                num_patch_clusters: int = 10, 
                preload: bool = False, 
                shuffle: bool = False, 
                patch_random: bool = False,
                load_adj_mat: bool = True):
        """
        Dataset for WSI features with clustering and graph data.
        Uses CSV columns to locate clustering files directly.
        
        Args:
            data_csv: Path to CSV file containing WSI information with clustering paths
            indices: Subset of indices to use
            graph_level: 'patch' or 'region' level processing
            num_patch_clusters: Number of clusters
            preload: Whether to preload all data
            shuffle: Whether to shuffle data
            patch_random: Whether to randomize patch order
            load_adj_mat: Whether to load adjacency matrices
        """
        super().__init__(data_csv, indices, preload=preload, shuffle=shuffle, patch_random=patch_random)
        
        if graph_level not in SUPPORTED_GRAPH_LEVELS:
            raise ValueError(f"graph_level must be one of {SUPPORTED_GRAPH_LEVELS}, got {graph_level}")
            
        self.graph_level = graph_level
        self.num_patch_clusters = num_patch_clusters
        self.load_adj_mat = load_adj_mat
        
        # Validate required columns exist in CSV
        self._validate_csv_columns()
        
        # Set adjacency matrix key
        self.adj_mat_key = 'patch_adj_mat' if graph_level == 'patch' else 'region_adj_mat'
        
        # Caches for derived data
        self.patch_cluster_indices_flat_cache: Dict[str, np.ndarray] = {}
        self.patch_adj_mats_cache: Dict[str, np.ndarray] = {}

        if self.preload and self.indices:
            logger.info(f"Preloading clustering data for WSIWithCluster (graph_level='{self.graph_level}')...")
            for case_id in tqdm(self.indices, desc=f"Preloading clustering ({self.graph_level})"):
                self._preload_single_wsi_derived_data(case_id)
            logger.info("WSIWithCluster preloading complete.")

    def _validate_csv_columns(self):
        """Validate that required clustering columns exist in CSV."""
        required_cols = ['patch_cluster_filepath', 'region_cluster_filepath']
        missing_cols = [col for col in required_cols if col not in self.samples_df.columns]
        if missing_cols:
            raise ValueError(f"Required clustering columns missing from CSV: {missing_cols}")

    def _get_clustering_path(self, case_id: str) -> Path:
        """Get the clustering file path for a case based on graph level."""
        if self.graph_level == 'patch':
            return Path(self.samples_df.at[case_id, 'patch_cluster_filepath'])
        else:  # region
            return Path(self.samples_df.at[case_id, 'region_cluster_filepath'])

    def _preload_single_wsi_derived_data(self, case_id: str):
        """Preload clustering data for a single WSI."""
        clustering_file_path = self._get_clustering_path(case_id)
        patch_cluster_indices_flat = np.array([], dtype=int)
        patch_adj_mat = np.array([])

        if clustering_file_path.exists():
            try:
                with np.load(clustering_file_path) as p_data:
                    # Load cluster assignments
                    cluster_key = 'patch_clusters' if self.graph_level == 'patch' else 'region_clusters'
                    if cluster_key in p_data:
                        patch_cluster_indices_flat = p_data[cluster_key].flatten()
                    elif 'features_cluster_indices' in p_data:  # Fallback key
                        patch_cluster_indices_flat = p_data['features_cluster_indices'].flatten()
                    else:
                        logger.warning(f"No cluster data found in {clustering_file_path}")
                    
                    # Load adjacency matrix if requested
                    if self.load_adj_mat and self.adj_mat_key in p_data:
                        patch_adj_mat = p_data[self.adj_mat_key]
                        
            except Exception as e:
                logger.error(f"Error preloading clustering data for {case_id} from {clustering_file_path}: {e}")
        else:
            logger.warning(f"Clustering file not found for {case_id} at {clustering_file_path}")
        
        self.patch_cluster_indices_flat_cache[case_id] = patch_cluster_indices_flat
        self.patch_adj_mats_cache[case_id] = patch_adj_mat

    def _load_clustering_data(self, case_id: str) -> Tuple[Optional[np.ndarray], Optional[torch.Tensor]]:
        """Load clustering data for a specific case."""
        clustering_file_path = self._get_clustering_path(case_id)
        
        if not clustering_file_path.exists():
            logger.warning(f"Clustering file not found: {clustering_file_path}")
            return None, None
        
        try:
            data = np.load(str(clustering_file_path))
            
            # Load cluster assignments
            cluster_key = 'patch_clusters' if self.graph_level == 'patch' else 'region_clusters'
            clusters = None
            if cluster_key in data:
                clusters = data[cluster_key]
            elif 'features_cluster_indices' in data:  # Fallback key
                clusters = data['features_cluster_indices']
            else:
                logger.warning(f"No cluster data found with keys '{cluster_key}' or 'features_cluster_indices' in {clustering_file_path}")
                return None, None
            
            # Load adjacency matrix if requested
            adj_mat = None
            if self.load_adj_mat and self.adj_mat_key in data:
                adj_mat = data[self.adj_mat_key]
                adj_mat = torch.from_numpy(adj_mat).float()
            
            return clusters, adj_mat
            
        except Exception as e:
            logger.error(f"Failed to load clustering data from {clustering_file_path}: {e}")
            return None, None

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, List[List[int]], Optional[torch.Tensor], torch.Tensor, torch.Tensor, str]:
        if not self.indices or index >= len(self.indices):
            raise IndexError(f"Invalid index {index} for dataset size {len(self.indices)}")
            
        case_id = self.indices[index]
        label = torch.tensor(self.samples_df.at[case_id, 'label'], dtype=torch.long)
        
        # =================================================================
        # START: SIMPLIFIED AND ROBUST DATA LOADING
        # =================================================================
        
        # The consolidated file path is the single source of truth.
        consolidated_npz_path = self._get_clustering_path(case_id)

        try:
            if not consolidated_npz_path.exists():
                raise FileNotFoundError(f"Consolidated data file not found for {case_id} at: {consolidated_npz_path}")

            with np.load(consolidated_npz_path) as data:
                # Load features and coords from the consolidated file
                features_np = data['img_features']
                coords_np = data.get('coords', np.zeros((features_np.shape[0], 2), dtype=np.float32))
                
                # Load cluster assignments
                clusters_np = data['patch_clusters'].flatten()
                
                # Load adjacency matrix
                adj_mat_np = data.get(self.adj_mat_key)

            # --- Data Validation ---
            if features_np.shape[0] != clusters_np.shape[0]:
                raise ValueError(f"Data inconsistency within {consolidated_npz_path}: "
                                 f"Found {features_np.shape[0]} features but {clusters_np.shape[0]} cluster assignments.")

            if adj_mat_np is not None and adj_mat_np.shape[0] != features_np.shape[0]:
                 raise ValueError(f"Adjacency matrix shape mismatch in {consolidated_npz_path}: "
                                 f"Matrix is {adj_mat_np.shape}, expected ({features_np.shape[0]}, {features_np.shape[0]})")

            # Convert to Tensors
            features_out = torch.as_tensor(features_np, dtype=torch.float32)
            coords_out = torch.as_tensor(coords_np, dtype=torch.float32)
            adj_mat_out = torch.as_tensor(adj_mat_np, dtype=torch.float32) if adj_mat_np is not None else None
            
        except Exception as e:
            logger.error(f"Failed to load or process data for {case_id} from {consolidated_npz_path}: {e}")
            # Return empty/dummy tensors on failure
            empty_feat = torch.empty(0, self.patch_dim, dtype=torch.float32)
            empty_coords = torch.empty(0, 2, dtype=torch.float32)
            return (empty_feat, [], None, empty_coords, label, case_id)

        # Reconstruct List[List[int]] for cluster_info
        cluster_info_list_of_lists: List[List[int]] = [[] for _ in range(self.num_patch_clusters)]
        for patch_idx, cluster_id in enumerate(clusters_np):
            if 0 <= cluster_id < self.num_patch_clusters:
                cluster_info_list_of_lists[cluster_id].append(patch_idx)
        
        # Fallback for adjacency matrix if it wasn't loaded but was expected
        if self.load_adj_mat and adj_mat_out is None:
            adj_mat_out = torch.eye(features_out.shape[0], dtype=torch.float32)
        
        return features_out, cluster_info_list_of_lists, adj_mat_out, coords_out, label, case_id


def mixup(inputs: torch.Tensor, alpha: Union[float, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Mix-up a batch tensor."""
    batch_size = inputs.shape[0]
    lambda_ = alpha + torch.rand(size=(batch_size, 1), device=inputs.device) * (1 - alpha)
    rand_idx = torch.randperm(batch_size, device=inputs.device)
    a = torch.stack([lambda_[i] * inputs[i] for i in range(batch_size)])
    b = torch.stack([(1 - lambda_[i]) * inputs[rand_idx[i]] for i in range(batch_size)])
    outputs = a + b
    return outputs, lambda_, rand_idx


def get_selected_bag_and_graph(feat_list, cluster_list, adj_mat_list, action_sequence, feat_size, device):
    """
    Final corrected version. Creates correctly re-indexed edge_index for subgraphs.
    This function is the single source of truth for creating model inputs.
    """
    batch_size = len(feat_list)
    if batch_size == 0:
        # Assuming DEFAULT_FEATURE_DIM is defined in this file
        return torch.empty(0, feat_size, DEFAULT_FEATURE_DIM, device=device), []

    selected_features_list = []
    # We will now return the correctly re-indexed edge_index for the GAT layer
    selected_edge_indices_list = []

    for i in range(batch_size):
        feat = feat_list[i]
        cluster = cluster_list[i]
        adj_mat = adj_mat_list[i]
        action = action_sequence[i]

        num_patches, feat_dim = feat.shape

        selected_indices_global = translate_ppo_action_to_patch_indices(
            action, cluster, feat_size, device
        )
        
        valid_mask = (selected_indices_global != -1) & (selected_indices_global < num_patches)
        final_indices_global = selected_indices_global[valid_mask][:feat_size]

        if final_indices_global.numel() == 0 or num_patches == 0:
            # Handle case with no valid patches
            selected_features = torch.zeros(feat_size, feat_dim, device=device, dtype=feat.dtype)
            edge_index = torch.empty((2, 0), dtype=torch.long, device=device)
        else:
            # Step 1: Select the features for the subgraph using global indices
            selected_feat = feat[final_indices_global]
            
            # Step 2: Create the subgraph's adjacency matrix
            sub_adj_mat = adj_mat[final_indices_global, :][:, final_indices_global]
            
            # Step 3: CRITICAL - Convert the subgraph adj matrix to a LOCAL edge index
            # The indices will now correctly be in the range [0, num_selected_nodes-1]
            edge_index = sub_adj_mat.nonzero(as_tuple=False).t().contiguous()
            
            # Step 4: Pad features if necessary to match the fixed bag size
            num_selected_nodes = selected_feat.shape[0]
            if num_selected_nodes < feat_size:
                padding = torch.zeros(feat_size - num_selected_nodes, feat_dim, device=device, dtype=selected_feat.dtype)
                selected_features = torch.cat([selected_feat, padding], dim=0)
            else:
                selected_features = selected_feat
        
        selected_features_list.append(selected_features)
        selected_edge_indices_list.append(edge_index)

    # Stack all processed features into a single batch tensor
    batched_features = torch.stack(selected_features_list, dim=0)
    
    # Return the list of correctly re-indexed edge_indices
    return batched_features, selected_edge_indices_list


def collate_mil_graph_batch(
    batch: List[Tuple[torch.Tensor, List[List[int]], Optional[torch.Tensor], torch.Tensor, torch.Tensor, str]]
) -> Dict[str, Union[torch.Tensor, List[str], List[Optional[torch.Tensor]], List[List[List[int]]]]]:
    """Collate function for WSIWithCluster batches."""
    if not batch:
        return {}

    features_list, cluster_list, adj_mats_list, coords_list, labels_list, case_ids_list = zip(*batch)
    
    device = features_list[0].device if features_list and features_list[0].numel() > 0 else torch.device('cpu')

    max_bag_size = max(f.shape[0] for f in features_list)
    
    d_feat = 0
    for f in features_list:
        if f.numel() > 0 and f.ndim == 2: 
            d_feat = f.shape[1]
            break
    if d_feat == 0:
        logger.warning("Collate: Could not infer feature dimension. Using fallback 1024.")
        d_feat = DEFAULT_FEATURE_DIM

    padded_features_batch = torch.zeros(len(batch), max_bag_size, d_feat, dtype=torch.float32, device=device)
    masks_batch = torch.zeros(len(batch), max_bag_size, dtype=torch.bool, device=device)

    for i, features_i in enumerate(features_list):
        num_instances_i = features_i.shape[0]
        if num_instances_i > 0:
            if features_i.shape[1] == d_feat:
                 padded_features_batch[i, :num_instances_i, :] = features_i
            else:
                 logger.warning(f"Collate: Feature dimension mismatch for item {i}. Expected {d_feat}, got {features_i.shape[1]}.")
            masks_batch[i, :num_instances_i] = True
    
    labels_tensor_list = [l if isinstance(l, torch.Tensor) else torch.tensor(l, device=device) for l in labels_list]
    labels_tensor = torch.stack(labels_tensor_list)

    return {
        'features': padded_features_batch,
        'clusters': list(cluster_list),
        'adj_mats': list(adj_mats_list),
        'coords': list(coords_list),
        'mask': masks_batch,
        'label': labels_tensor,
        'case_id': list(case_ids_list)
    }