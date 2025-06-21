"""
Minimal, robust WSI dataset implementation for SG-MuRCL pipeline.
Focused on consolidated .npz files with clustered/graph data.
"""

import logging
import random
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

logger = logging.getLogger(__name__)

DEFAULT_FEATURE_DIM = 1024  # Keep as 1024 as requested


class SGMuRCLDataset(Dataset):
    """
    Simplified WSI dataset for SG-MuRCL pipeline.
    Loads consolidated .npz files containing features, clusters, adjacency matrices, and coordinates.
    """

    def __init__(
        self,
        data_csv: Union[str, Path],
        indices: Optional[Iterable[str]] = None,
        num_clusters: int = 10,
        preload: bool = False,
        shuffle: bool = False,
        load_adj_mat: bool = True,
    ) -> None:
        """
        Initialize dataset.
        
        Args:
            data_csv: Path to CSV with columns: case_id, label, features_filepath, patch_cluster_filepath
            indices: Subset of case_ids to use (None = use all)
            num_clusters: Expected number of clusters in data
            preload: Whether to preload data into memory
            shuffle: Whether to shuffle indices
            load_adj_mat: Whether to load adjacency matrices
        """
        super().__init__()
        
        self.data_csv_path = Path(data_csv)
        self.num_clusters = num_clusters
        self.preload = preload
        self.load_adj_mat = load_adj_mat
        
        # Load and validate CSV
        self.samples_df = self._load_csv()
        
        # Create data_dict for compatibility
        self.data_dict = {}
        for case_id, row in self.samples_df.iterrows():
            self.data_dict[case_id] = {
                'data_path': Path(row['data_filepath']),
                'label': row['label']
            }
        
        # Set indices
        if indices is not None:
            self.indices = [idx for idx in indices if idx in self.samples_df.index]
            if len(self.indices) != len(list(indices)):
                logger.warning("Some provided indices were not found in CSV")
        else:
            self.indices = self.samples_df.index.tolist()
        
        if not self.indices:
            logger.warning("No valid indices found. Dataset will be empty.")
            self.feature_dim = DEFAULT_FEATURE_DIM
            return
        
        if shuffle:
            random.shuffle(self.indices)
            
        # Determine feature dimension
        self.feature_dim = self._determine_feature_dim()
        
        # Preload data if requested
        self.data_cache: Dict[str, Dict] = {}
        if self.preload:
            self._preload_data()

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, List[List[int]], torch.Tensor, str, torch.Tensor, torch.Tensor]:
        """
        Get item from dataset.
        
        Returns:
            features: [N, D] patch features
            clusters: List of lists, where clusters[i] contains patch indices for cluster i
            label: WSI label
            case_id: Case identifier
            adj_mat: [N, N] adjacency matrix or None
            coords: [N, 2] patch coordinates
        """
        if index >= len(self.indices):
            raise IndexError(f"Index {index} out of range for dataset size {len(self.indices)}")
        
        case_id = self.indices[index]
        label = torch.tensor(self.samples_df.at[case_id, 'label'], dtype=torch.long)
        
        # Load data
        if self.preload and case_id in self.data_cache:
            data = self.data_cache[case_id]
        else:
            data = self._load_case_data(case_id)
        
        # Extract components
        features = torch.as_tensor(data['features'], dtype=torch.float32)
        coords = torch.as_tensor(data['coords'], dtype=torch.float32)
        adj_mat = torch.as_tensor(data['adj_mat'], dtype=torch.float32) if data['adj_mat'] is not None else None
        
        # Build cluster lists
        clusters = self._build_cluster_lists(data['cluster_labels'])
        
        return features, clusters, label, case_id, adj_mat, coords

    def _load_csv(self) -> pd.DataFrame:
        """Load and validate CSV file."""
        try:
            df = pd.read_csv(self.data_csv_path)
            required_cols = ['case_id', 'label']
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                raise ValueError(f"Missing required columns: {missing_cols}")
            
            # Check which filepath column to use
            if 'patch_cluster_filepath' in df.columns:
                logger.info("Using patch_cluster_filepath (consolidated data with clustering)")
                df['data_filepath'] = df['patch_cluster_filepath']
            elif 'features_filepath' in df.columns:
                logger.info("Using features_filepath (raw features without clustering)")
                df['data_filepath'] = df['features_filepath']
            else:
                raise ValueError("Neither 'patch_cluster_filepath' nor 'features_filepath' found in CSV")
            
            df.set_index('case_id', inplace=True, drop=False)
            return df
            
        except Exception as e:
            logger.error(f"Error loading CSV {self.data_csv_path}: {e}")
            raise

    def _determine_feature_dim(self) -> int:
        """Determine feature dimension from first valid case."""
        for case_id in self.indices[:5]:  # Check first few cases
            try:
                data = self._load_case_data(case_id)
                if data['features'].size > 0:
                    actual_dim = data['features'].shape[1]
                    logger.info(f"Detected feature dimension: {actual_dim}")
                    return actual_dim
            except Exception as e:
                logger.warning(f"Error checking feature dim for {case_id}: {e}")
                continue
        
        logger.warning("Could not determine feature dimension, using default")
        return DEFAULT_FEATURE_DIM

    def _load_case_data(self, case_id: str) -> Dict:
        """Load and validate case data using CORRECT field names from .npz files."""
        data_path = self.data_dict[case_id]['data_path']
        
        try:
            with np.load(data_path) as data:
                # Print available keys for debugging (remove in production)
                logger.debug(f"Available keys in {case_id}: {list(data.keys())}")
                
                # Features - use 'img_features' instead of 'features'
                features = data.get('img_features')
                if features is None:
                    # Fallback to 'features' if 'img_features' not found
                    features = data.get('features')
                
                if features is None or features.size == 0:
                    logger.warning(f"No features found for {case_id}")
                    return self._get_empty_case_data()
                
                features = features.astype(np.float32)
                
                # Get number of patches from features
                if features.ndim == 1:
                    num_patches = 1
                    features = features.reshape(1, -1)
                else:
                    num_patches = features.shape[0]
                
                # Log actual feature dimensions for debugging
                logger.debug(f"Loaded features for {case_id}: shape {features.shape}")
                
                # Validate that features match expected number of patches
                if features.shape[0] != num_patches:
                    logger.warning(f"Feature count mismatch for {case_id}: got {features.shape[0]}, expected {num_patches}")
                    num_patches = features.shape[0]
                
                # Coordinates - already correctly named 'coords'
                coords = data.get('coords', np.zeros((num_patches, 2), dtype=np.float32))
                if coords.shape[0] != num_patches:
                    logger.warning(f"Coord/feature mismatch for {case_id}, using zero coords")
                    coords = np.zeros((num_patches, 2), dtype=np.float32)
                
                # Cluster labels - already correctly named 'patch_clusters'
                cluster_labels = data.get('patch_clusters')
                if cluster_labels is None:
                    logger.warning(f"No cluster labels for {case_id}, assigning to cluster 0")
                    cluster_labels = np.zeros(num_patches, dtype=np.int32)
                else:
                    cluster_labels = cluster_labels.astype(np.int32)
                    if len(cluster_labels) != num_patches:
                        logger.warning(f"Cluster/feature mismatch for {case_id}: {len(cluster_labels)} vs {num_patches}")
                        cluster_labels = np.zeros(num_patches, dtype=np.int32)
                
                # Adjacency matrix - use 'patch_adj_mat' instead of 'adj_mat'
                adj_mat = None
                if self.load_adj_mat:
                    adj_mat_raw = data.get('patch_adj_mat')
                    if adj_mat_raw is None:
                        # Fallback to 'adj_mat' if 'patch_adj_mat' not found
                        adj_mat_raw = data.get('adj_mat')
                    
                    if adj_mat_raw is not None:
                        adj_mat_raw = adj_mat_raw.astype(np.float32)
                        
                        # Validate adjacency matrix dimensions and properties
                        if adj_mat_raw.ndim == 2 and adj_mat_raw.shape[0] == num_patches and adj_mat_raw.shape[1] == num_patches:
                            # Check if it's a valid adjacency matrix
                            if np.all(np.isfinite(adj_mat_raw)) and np.all(adj_mat_raw >= 0):
                                adj_mat = adj_mat_raw
                                logger.debug(f"Loaded adjacency matrix for {case_id}: shape {adj_mat.shape}")
                            else:
                                logger.warning(f"Invalid adj_mat values for {case_id} (contains inf/nan/negative), ignoring")
                                adj_mat = None
                        elif adj_mat_raw.ndim == 0:
                            logger.warning(f"Invalid adj_mat for {case_id}: 0-dimensional tensor, ignoring")
                            adj_mat = None
                        elif adj_mat_raw.ndim == 1:
                            logger.warning(f"Invalid adj_mat for {case_id}: 1-dimensional array with shape {adj_mat_raw.shape}, ignoring")
                            adj_mat = None
                        else:
                            logger.warning(f"Invalid adj_mat shape for {case_id}: {adj_mat_raw.shape} vs ({num_patches}, {num_patches}), ignoring")
                            adj_mat = None
                    else:
                        logger.debug(f"No adjacency matrix found for {case_id}")
                
                return {
                    'features': features,
                    'coords': coords,
                    'cluster_labels': cluster_labels,
                    'adj_mat': adj_mat
                }
                
        except Exception as e:
            logger.error(f"Error loading data for {case_id}: {e}")
            return self._get_empty_case_data()
    
    def _get_empty_case_data(self) -> Dict:
        """Return empty/default data structure."""
        return {
            'features': np.zeros((0, self.feature_dim), dtype=np.float32),
            'coords': np.zeros((0, 2), dtype=np.float32),
            'cluster_labels': np.array([], dtype=np.int32),
            'adj_mat': None
        }

    def _build_cluster_lists(self, cluster_labels: np.ndarray) -> List[List[int]]:
        """Build list of lists where clusters[i] contains patch indices for cluster i."""
        clusters = [[] for _ in range(self.num_clusters)]
        
        for patch_idx, cluster_id in enumerate(cluster_labels):
            if 0 <= cluster_id < self.num_clusters:
                clusters[cluster_id].append(patch_idx)
            else:
                # Invalid cluster ID, assign to cluster 0
                clusters[0].append(patch_idx)
        
        return clusters

    def _preload_data(self) -> None:
        """Preload all data into memory."""
        logger.info(f"Preloading data for {len(self.indices)} cases...")
        
        for case_id in tqdm(self.indices, desc="Preloading"):
            self.data_cache[case_id] = self._load_case_data(case_id)
        
        logger.info("Preloading complete")

    def shuffle(self) -> None:
        """Shuffle dataset indices."""
        random.shuffle(self.indices)


# ... rest of the file remains the same ...


def translate_ppo_action_to_patch_indices(
    ppo_cluster_action: torch.Tensor,
    patch_clusters_indices: List[List[int]],
    num_total_patches_to_select: int,
    device: torch.device
) -> torch.Tensor:
    """
    Translate PPO cluster actions to specific patch indices.
    
    Args:
        ppo_cluster_action: [num_clusters] action values for each cluster
        patch_clusters_indices: List of lists, where each list contains patch indices for a cluster
        num_total_patches_to_select: Target number of patches to select
        device: Device for tensors
        
    Returns:
        Tensor of selected patch indices, with -1 for padding
    """
    selected_patch_indices_list = []
    num_actual_clusters_for_wsi = len(patch_clusters_indices)
    
    if num_actual_clusters_for_wsi == 0 or num_total_patches_to_select <= 0:
        return torch.full((num_total_patches_to_select,), -1, dtype=torch.long, device=device)
    
    # Use only the relevant actions for this WSI
    current_ppo_action = ppo_cluster_action[:num_actual_clusters_for_wsi]
    
    # Filter out empty clusters
    valid_cluster_indices = [i for i, patches in enumerate(patch_clusters_indices) if patches]
    
    if not valid_cluster_indices:
        return torch.full((num_total_patches_to_select,), -1, dtype=torch.long, device=device)
    
    # Get actions and patch lists for valid clusters
    valid_actions = current_ppo_action[valid_cluster_indices]
    valid_cluster_patches = [patch_clusters_indices[i] for i in valid_cluster_indices]
    
    if valid_actions.numel() == 0:
        return torch.full((num_total_patches_to_select,), -1, dtype=torch.long, device=device)
    
    # Convert actions to probabilities
    cluster_probs = torch.softmax(valid_actions, dim=0)
    
    # Handle numerical issues
    if not torch.isfinite(cluster_probs).all() or cluster_probs.sum() == 0:
        cluster_probs = torch.ones(len(valid_cluster_patches), device=device) / len(valid_cluster_patches)
    
    try:
        # Sample clusters according to probabilities
        sampled_cluster_indices = torch.multinomial(
            cluster_probs, num_total_patches_to_select, replacement=True
        )
    except RuntimeError:
        # Fallback to uniform sampling
        uniform_probs = torch.ones(len(valid_cluster_patches), device=device) / len(valid_cluster_patches)
        sampled_cluster_indices = torch.multinomial(uniform_probs, num_total_patches_to_select, replacement=True)
    
    # Select patches from chosen clusters
    for cluster_idx in sampled_cluster_indices:
        cluster_patches = valid_cluster_patches[cluster_idx]
        if cluster_patches:
            selected_patch = random.choice(cluster_patches)
            selected_patch_indices_list.append(selected_patch)
        else:
            selected_patch_indices_list.append(-1)
    
    return torch.tensor(selected_patch_indices_list, dtype=torch.long, device=device)


def get_selected_bag_and_graph(
    feat_list: List[torch.Tensor],
    cluster_list: List[List[List[int]]],
    adj_mat_list: List[Optional[torch.Tensor]],
    action_sequence_batch: torch.Tensor,
    feat_size: int,
    device: torch.device
) -> Tuple[List[torch.Tensor], List[Optional[torch.Tensor]], List[torch.Tensor]]:
    """
    Create selected bags of features with corresponding adjacency matrices and masks.
    
    Args:
        feat_list: List of feature tensors [N_i, D] for each WSI
        cluster_list: List of cluster assignments for each WSI
        adj_mat_list: List of adjacency matrices [N_i, N_i] for each WSI
        action_sequence_batch: Batch of action sequences [B, num_clusters]
        feat_size: Target bag size
        device: Device for tensors
        
    Returns:
        Tuple of (selected_features, selected_adj_mats, selected_masks)
    """
    batch_size = len(feat_list)
    if batch_size == 0:
        return [], [], []
    
    selected_features_list = []
    selected_adj_mats_list = []
    selected_masks_list = []
    
    for i in range(batch_size):
        features = feat_list[i]
        clusters = cluster_list[i]
        adj_mat = adj_mat_list[i]
        actions = action_sequence_batch[i]
        
        # Validate inputs
        if not isinstance(features, torch.Tensor) or features.ndim != 2:
            logger.error(f"Invalid features for WSI {i}")
            feat_dim = DEFAULT_FEATURE_DIM
            selected_features_list.append(torch.zeros((feat_size, feat_dim), device=device, dtype=torch.float32))
            selected_adj_mats_list.append(None)
            selected_masks_list.append(torch.zeros(feat_size, device=device, dtype=torch.bool))
            continue
        
        num_patches = features.shape[0]
        feat_dim = features.shape[1]
        
        if num_patches == 0:
            selected_features_list.append(torch.zeros((feat_size, feat_dim), device=device, dtype=features.dtype))
            selected_adj_mats_list.append(None)
            selected_masks_list.append(torch.zeros(feat_size, device=device, dtype=torch.bool))
            continue
        
        # Validate adjacency matrix - ADD THIS VALIDATION
        valid_adj_mat = None
        if adj_mat is not None and isinstance(adj_mat, torch.Tensor):
            # Check if adjacency matrix has correct dimensions
            if adj_mat.ndim == 2 and adj_mat.shape[0] == num_patches and adj_mat.shape[1] == num_patches:
                valid_adj_mat = adj_mat
            elif adj_mat.ndim == 0:
                # Handle 0-dimensional tensor (scalar) - treat as no adjacency matrix
                logger.warning(f"WSI {i}: Adjacency matrix is 0-dimensional, treating as None")
                valid_adj_mat = None
            elif adj_mat.ndim == 1:
                # Handle 1-dimensional tensor - treat as no adjacency matrix
                logger.warning(f"WSI {i}: Adjacency matrix is 1-dimensional, treating as None")
                valid_adj_mat = None
            else:
                # Handle incorrect shape
                logger.warning(f"WSI {i}: Adjacency matrix shape {adj_mat.shape} doesn't match features {num_patches}, treating as None")
                valid_adj_mat = None
        
        # Select patch indices based on actions
        selected_indices = translate_ppo_action_to_patch_indices(actions, clusters, feat_size, device)
        
        # Initialize outputs
        bag_features = torch.zeros((feat_size, feat_dim), device=device, dtype=features.dtype)
        bag_mask = torch.zeros(feat_size, device=device, dtype=torch.bool)
        
        # Fill valid positions
        valid_mask = (selected_indices != -1) & (selected_indices < num_patches)
        valid_positions = torch.where(valid_mask)[0]
        valid_indices = selected_indices[valid_positions]
        
        if valid_positions.numel() > 0:
            bag_features[valid_positions] = features[valid_indices]
            bag_mask[valid_positions] = True
        
        # Create sub-adjacency matrix - USE VALIDATED ADJ_MAT
        bag_adj_mat = None
        if valid_positions.numel() > 0 and valid_adj_mat is not None:
            try:
                # Extract sub-matrix
                sub_adj = valid_adj_mat[valid_indices][:, valid_indices]
                
                # Create padded adjacency matrix
                bag_adj_mat = torch.zeros((feat_size, feat_size), device=device, dtype=torch.float32)
                bag_adj_mat[valid_positions.unsqueeze(1), valid_positions] = sub_adj
            except Exception as e:
                logger.warning(f"WSI {i}: Failed to create sub-adjacency matrix: {e}. Using None.")
                bag_adj_mat = None
        
        selected_features_list.append(bag_features)
        selected_adj_mats_list.append(bag_adj_mat)
        selected_masks_list.append(bag_mask)
    
    return selected_features_list, selected_adj_mats_list, selected_masks_list


def collate_mil_graph_batch(
    batch: List[Tuple[torch.Tensor, List[List[int]], torch.Tensor, str, Optional[torch.Tensor], torch.Tensor]]
) -> Dict[str, Union[torch.Tensor, List]]:
    """
    Collate function for batching WSI data, matching SmMIL expectations.
    """
    if not batch:
        return {}
    
    features_list, cluster_list, label_list, case_id_list, adj_mat_list, coords_list = zip(*batch)
    
    # Determine device and dimensions
    device = features_list[0].device if features_list[0].numel() > 0 else torch.device('cpu')
    max_patches = max(f.shape[0] for f in features_list)
    
    # Determine feature dimension
    feat_dim = DEFAULT_FEATURE_DIM
    for f in features_list:
        if f.numel() > 0 and f.ndim == 2:
            feat_dim = f.shape[1]
            break
    
    # Create padded feature tensor
    padded_features = torch.zeros(len(batch), max_patches, feat_dim, dtype=torch.float32, device=device)
    masks = torch.zeros(len(batch), max_patches, dtype=torch.bool, device=device)  # Boolean mask!
    
    for i, features in enumerate(features_list):
        num_patches = features.shape[0]
        if num_patches > 0:
            if features.shape[1] == feat_dim:
                padded_features[i, :num_patches] = features
                masks[i, :num_patches] = True  # Set valid positions to True
            else:
                logger.warning(f"Feature dimension mismatch for batch item {i}")
    
    # Stack labels
    labels = torch.stack([torch.tensor(l) if not isinstance(l, torch.Tensor) else l for l in label_list])
    
    return {
        'features': padded_features,
        'clusters': list(cluster_list),
        'adj_mats': list(adj_mat_list),
        'coords': list(coords_list),
        'mask': masks,  # Boolean tensor [B, N]
        'label': labels,
        'case_id': list(case_id_list)
    }


def mixup(
    inputs: Union[torch.Tensor, List[torch.Tensor]], 
    alpha: Union[float, torch.Tensor]
) -> Tuple[Union[torch.Tensor, List[torch.Tensor]], torch.Tensor, torch.Tensor]:
    """
    Apply mixup augmentation to inputs.
    
    Args:
        inputs: Either a tensor [B, ...] or list of tensors
        alpha: Mixup parameter
        
    Returns:
        Tuple of (mixed_inputs, lambda_, rand_idx)
    """
    if isinstance(inputs, list):
        if len(inputs) == 0:
            return inputs, torch.tensor([]), torch.tensor([])
        
        batch_size = len(inputs)
        device = inputs[0].device if len(inputs) > 0 else torch.device('cpu')
        
        # Generate mixup parameters
        lambda_ = alpha + torch.rand(size=(batch_size, 1), device=device) * (1 - alpha)
        rand_idx = torch.randperm(batch_size, device=device)
        
        # Apply mixup
        mixed_inputs = []
        for i in range(batch_size):
            if i < len(inputs) and rand_idx[i] < len(inputs):
                mixed_tensor = lambda_[i] * inputs[i] + (1 - lambda_[i]) * inputs[rand_idx[i]]
                mixed_inputs.append(mixed_tensor)
            else:
                mixed_inputs.append(inputs[i] if i < len(inputs) else inputs[0])
        
        return mixed_inputs, lambda_, rand_idx
    
    else:
        # Handle tensor input
        if not isinstance(inputs, torch.Tensor):
            raise TypeError(f"Expected tensor or list of tensors, got {type(inputs)}")
        
        batch_size = inputs.shape[0]
        lambda_ = alpha + torch.rand(size=(batch_size, 1), device=inputs.device) * (1 - alpha)
        rand_idx = torch.randperm(batch_size, device=inputs.device)
        
        mixed_inputs = torch.stack([
            lambda_[i] * inputs[i] + (1 - lambda_[i]) * inputs[rand_idx[i]]
            for i in range(batch_size)
        ])
        
        return mixed_inputs, lambda_, rand_idx