import torch
import torch.nn as nn
from typing import Optional
import logging

logger = logging.getLogger(__name__)

try:
    from torch_geometric.nn import GATConv
except ImportError:
    GATConv = None
    logger.warning("PyTorch Geometric not found. GAT functionality will be unavailable.")

class BatchedGATWrapper(nn.Module):
    """
    A wrapper for GATConv that handles batched dense adjacency matrices.
    It accepts batch-first tensors and iterates internally to apply GAT to each graph.
    This is the standard way to wrap a non-batchable layer for a batch-first pipeline.
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

    def forward(self, features_batch: torch.Tensor, adj_mats_batch: Optional[torch.Tensor]) -> torch.Tensor:
        """
        Processes a batch of graphs using dense adjacency matrices.

        Args:
            features_batch (torch.Tensor): Batched node features of shape [B, N, D_in].
            adj_mats_batch (Optional[torch.Tensor]): Batched adjacency matrices of shape [B, N, N].
        
        Returns:
            torch.Tensor: Batched processed node features of shape [B, N, D_out].
        """
        if adj_mats_batch is None:
            logger.debug("No adjacency matrices provided to GAT; returning features unchanged.")
            # If the output dimension is different, we must project the features.
            if self.input_dim != self.output_dim:
                 # This case should be handled by a dedicated linear layer if needed,
                 # but for now we'll return as is and rely on downstream checks.
                 logger.warning("GAT input and output dimensions differ, but no adj_mats provided. Feature dimensions will be inconsistent.")
            return features_batch

        batch_size = features_batch.shape[0]
        outputs = []

        # Loop over each item in the batch
        for i in range(batch_size):
            features = features_batch[i]  # Shape: [N, D_in]
            adj_mat = adj_mats_batch[i]   # Shape: [N, N]

            # CRITICAL STEP: Convert dense adjacency matrix to sparse edge_index
            # GATConv expects a [2, num_edges] tensor.
            edge_index = torch.nonzero(adj_mat).t().contiguous()

            # Apply GAT layer
            processed_features = self.gat_layer(features, edge_index)
            processed_features = self.elu(processed_features)
            outputs.append(processed_features)

        # Stack the results from the loop back into a single batch tensor
        return torch.stack(outputs, dim=0)

def create_gat_encoder(input_dim: int, **kwargs) -> Optional[BatchedGATWrapper]:
    """Factory function to create a GAT encoder."""
    if GATConv is None:
        logger.error("Cannot create GAT encoder because PyTorch Geometric is not installed.")
        return None
    
    # Extract GAT-specific arguments from kwargs, providing defaults
    gat_args = {
        'output_dim': kwargs.get('gat_hidden_dim', 512),
        'n_heads': kwargs.get('gat_n_heads', 8),
        'dropout': kwargs.get('gat_dropout', 0.25)
    }
    
    logger.info(f"Creating GAT encoder with args: {gat_args}")
    return BatchedGATWrapper(input_dim=input_dim, **gat_args)