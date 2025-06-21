import torch
import torch.nn as nn
from typing import Optional, Tuple
import logging

logger = logging.getLogger(__name__)

try:
    from torch_geometric.nn import GATConv
    from torch_geometric.utils import dense_to_sparse
except ImportError:
    GATConv = None
    dense_to_sparse = None
    logger.warning("PyTorch Geometric not found. GAT functionality will be unavailable.")


class BatchedGATWrapper(nn.Module):
    """
    Wrapper for GAT that handles batched inputs with adjacency matrices.
    """
    
    def __init__(self, input_dim, output_dim, n_heads=8, dropout=0.25):
        super().__init__()
        if GATConv is None:
            raise ImportError("torch_geometric.nn.GATConv is required for BatchedGATWrapper.")
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.n_heads = n_heads
        
        # Ensure output_dim is divisible by n_heads for GATConv
        if output_dim % n_heads != 0:
            raise ValueError(f"GAT output_dim ({output_dim}) must be divisible by n_heads ({n_heads}).")
            
        self.gat_layer = GATConv(input_dim, output_dim // n_heads, heads=n_heads, dropout=dropout)
        self.elu = nn.ELU()

    def forward(self, 
                features_batch: torch.Tensor, 
                adj_mats_batch: Optional[torch.Tensor] = None,
                **kwargs) -> torch.Tensor:
        """
        Apply GAT to a batch of feature matrices.
        
        Args:
            features_batch: [B, N, D] batch of node features
            adj_mats_batch: [B, N, N] batch of adjacency matrices (optional)
            
        Returns:
            processed_features: [B, N, output_dim] processed features
        """
        if dense_to_sparse is None:
            raise ImportError("torch_geometric.utils.dense_to_sparse is required.")
            
        batch_size, num_nodes, feature_dim = features_batch.shape
        device = features_batch.device
        
        # Handle case where no adjacency matrices are provided
        if adj_mats_batch is None:
            # Create identity adjacency matrices (self-loops only)
            adj_mats_batch = torch.eye(num_nodes, device=device).unsqueeze(0).repeat(batch_size, 1, 1)
        
        processed_batch = []
        
        for i in range(batch_size):
            features = features_batch[i]  # [N, D]
            adj_mat = adj_mats_batch[i]   # [N, N]
            
            # Skip empty feature matrices
            if features.shape[0] == 0:
                empty_features = torch.zeros(0, self.output_dim, device=device, dtype=features.dtype)
                processed_batch.append(empty_features)
                continue
            
            try:
                # Convert dense adjacency matrix to edge_index format
                edge_index, edge_attr = dense_to_sparse(adj_mat)
                
                # Apply GAT layer
                processed_features = self.gat_layer(features, edge_index)
                processed_features = self.elu(processed_features)
                
                processed_batch.append(processed_features)
                
            except Exception as e:
                logger.warning(f"GAT processing failed for batch {i}: {e}. Using identity transformation.")
                # Fallback: linear transformation to match output dimension
                if not hasattr(self, '_fallback_linear'):
                    self._fallback_linear = nn.Linear(feature_dim, self.output_dim).to(device)
                processed_features = self._fallback_linear(features)
                processed_batch.append(processed_features)
        
        # Pad sequences to same length for batching
        max_nodes = max(f.shape[0] for f in processed_batch)
        if max_nodes == 0:
            return torch.zeros(batch_size, 0, self.output_dim, device=device, dtype=features_batch.dtype)
        
        padded_batch = []
        for features in processed_batch:
            if features.shape[0] < max_nodes:
                padding = torch.zeros(max_nodes - features.shape[0], self.output_dim, 
                                    device=device, dtype=features.dtype)
                padded_features = torch.cat([features, padding], dim=0)
            else:
                padded_features = features
            padded_batch.append(padded_features)
        
        return torch.stack(padded_batch)


def create_gat_encoder(input_dim: int, **kwargs) -> Optional[BatchedGATWrapper]:
    """Factory function to create a GAT encoder."""
    if GATConv is None:
        logger.error("PyTorch Geometric not available. Cannot create GAT encoder.")
        return None
    
    try:
        return BatchedGATWrapper(input_dim=input_dim, **kwargs)
    except Exception as e:
        logger.error(f"Failed to create GAT encoder: {e}")
        return None