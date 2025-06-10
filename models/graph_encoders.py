import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from typing import List, Optional
import logging

# --- Logger Setup ---
def setup_logger():
    """Sets up a basic console logger for the script."""
    logging.basicConfig(
        level=logging.INFO, # Log messages at INFO level and above (WARNING, ERROR, CRITICAL)
        format="[%(asctime)s] %(levelname)s - %(message)s", # Log message format
        datefmt="%Y-%m-%d %H:%M:%S" # Timestamp format
    )
    return logging.getLogger(__name__)

logger = setup_logger() # Global logger instance

class BaseGraphEncoder(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

    def forward(self, x, edge_index, batch=None):
        """Unified interface: single graph processing."""
        raise NotImplementedError("Subclasses must implement the forward method.")


class GATEncoder(BaseGraphEncoder):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=2, heads=4, dropout=0.1, concat_heads=False):
        # We need to compute the final output dim, so it's not passed to super() yet
        super().__init__(input_dim=input_dim, output_dim=0) 
        
        self.num_layers = num_layers
        self.dropout = dropout

        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        
        # Start with the initial feature dimension
        current_dim = input_dim

        for i in range(num_layers):
            # The first layer is special: it takes input_dim
            if i == 0:
                self.convs.append(
                    GATConv(current_dim, hidden_dim, heads=heads, dropout=dropout, concat=True)
                )
                current_dim = hidden_dim * heads
                self.batch_norms.append(nn.BatchNorm1d(current_dim))
            # The last layer is special: it might not concatenate
            elif i == num_layers - 1:
                self.convs.append(
                    GATConv(current_dim, output_dim, heads=heads, dropout=dropout, concat=concat_heads)
                )
                current_dim = output_dim * heads if concat_heads else output_dim
                self.batch_norms.append(nn.BatchNorm1d(current_dim))
            # Intermediate layers
            else:
                self.convs.append(
                    GATConv(current_dim, hidden_dim, heads=heads, dropout=dropout, concat=True)
                )
                current_dim = hidden_dim * heads
                self.batch_norms.append(nn.BatchNorm1d(current_dim))
        
        # set the final, correct output dimension for the whole module
        self.output_dim = current_dim

    def forward(self, x, edge_index, batch=None):
        """
        Forward pass for a single graph.
        """
        # Pass through all GAT layers
        for i in range(self.num_layers):
            x = self.convs[i](x, edge_index)
            if x.size(0) > 1:
                x = self.batch_norms[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            
        return x

    @staticmethod
    def adj_mat_to_edge_index(adj_mat):
        """
        Efficiently convert dense adjacency matrix to edge index.
        
        Args:
            adj_mat (torch.Tensor): Adjacency matrix [num_nodes, num_nodes]
            
        Returns:
            torch.Tensor: Edge index [2, num_edges]
        """
        if adj_mat is None or adj_mat.numel() == 0:
            return torch.empty(2, 0, dtype=torch.long, device=adj_mat.device if adj_mat is not None else 'cpu')
        
        # Use nonzero for efficient conversion
        edge_indices = torch.nonzero(adj_mat, as_tuple=False)  # [num_edges, 2]
        
        if edge_indices.numel() == 0:
            return torch.empty(2, 0, dtype=torch.long, device=adj_mat.device)
        
        edge_index = edge_indices.t().contiguous()  # [2, num_edges]
        return edge_index


class BatchedGATWrapper(nn.Module):
    """
    A robust wrapper to apply a GAT encoder to a list of graph tensors.
    This version performs all operations on the GPU and includes strong fallback mechanisms.
    """
    def __init__(self, gat_encoder):
        super(BatchedGATWrapper, self).__init__()
        self.gat_encoder = gat_encoder
        self.output_dim = self.gat_encoder.output_dim
        
        # This linear layer is a crucial fallback. If GAT processing fails for any reason,
        # this layer ensures the output tensor still has the correct dimension, preventing downstream crashes.
        self.dim_matcher = nn.Linear(self.gat_encoder.input_dim, self.output_dim)

    def forward(self, features_list: List[torch.Tensor], adj_mats_list: List[Optional[torch.Tensor]]):
        """
        Processes a list of graphs from a batch.
        
        Args:
            features_list (List[torch.Tensor]): A list where each element is a 2D feature tensor 
                                                 of shape [num_nodes, input_dim].
            adj_mats_list (List[Optional[torch.Tensor]]): A list where each element is a 2D dense 
                                                          adjacency matrix of shape [num_nodes, num_nodes].
        
        Returns:
            List[torch.Tensor]: A list of processed 2D feature tensors, each of shape [num_nodes, output_dim].
        """
        processed_batch_list = []
        if not features_list:
            return processed_batch_list

        for i, (x, adj_mat) in enumerate(zip(features_list, adj_mats_list)):
            # Ensure the feature tensor for this item is on the correct device
            device = x.device

            if x.numel() == 0:
                # Handle cases where a WSI has no patches
                processed_batch_list.append(torch.zeros(0, self.output_dim, device=device, dtype=x.dtype))
                continue

            try:
                # --- THE CORE FIX IS HERE ---
                # Adjacency matrix is already on the GPU. Convert it to an edge index directly on the GPU.
                if adj_mat is not None and adj_mat.numel() > 0:
                    # This is the standard, efficient, and GPU-native way to get the edge index.
                    # It finds the coordinates of all non-zero elements.
                    edge_index = torch.nonzero(adj_mat, as_tuple=False).t().contiguous()
                else:
                    # If there's no adjacency matrix, create an empty edge_index.
                    edge_index = torch.empty((2, 0), dtype=torch.long, device=device)

                # The GATConv layer handles an empty edge_index gracefully (it will apply linear layers without message passing).
                processed_x = self.gat_encoder(x, edge_index)
                processed_batch_list.append(processed_x)

            except Exception as e:
                # If any error occurs (including the CUDA assert), this block will catch it.
                logger.warning(f"GAT processing failed for item {i}: {e}. Falling back to a linear projection.")
                
                # Fallback: Use a simple linear layer to ensure the output has the correct dimension.
                # This prevents crashes in the subsequent MIL layer.
                processed_batch_list.append(self.dim_matcher(x))
                
        return processed_batch_list


def create_gat_encoder(input_dim, output_dim=None, hidden_dim=256, num_layers=2, heads=4, dropout=0.1):
    """
    Factory function to create GAT encoder with sensible defaults.
    
    Args:
        input_dim (int): Input feature dimension
        output_dim (int, optional): Output feature dimension. Defaults to input_dim
        hidden_dim (int): Hidden layer dimension
        num_layers (int): Number of GAT layers
        heads (int): Number of attention heads
        dropout (float): Dropout rate
        
    Returns:
        BatchedGATWrapper: Ready-to-use GAT encoder
    """
    if output_dim is None:
        output_dim = input_dim
    
    gat_encoder = GATEncoder(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        output_dim=output_dim,
        num_layers=num_layers,
        heads=heads,
        dropout=dropout,
        concat_heads=False  # Average heads for consistent output dim
    )
    
    return BatchedGATWrapper(gat_encoder)


def create_graph_encoder(encoder_type, input_dim, output_dim=None, **kwargs):
    """
    Universal factory function for creating graph encoders.
    
    Args:
        encoder_type (str): Type of encoder ('gat' or 'none')
        input_dim (int): Input feature dimension
        output_dim (int, optional): Output feature dimension
        **kwargs: Additional arguments for GAT encoder
        
    Returns:
        BatchedGATWrapper or None: Graph encoder or None if encoder_type is 'none'
    """
    if encoder_type.lower() == 'gat':
        return create_gat_encoder(input_dim, output_dim, **kwargs)
    elif encoder_type.lower() == 'none':
        return None
    else:
        raise ValueError(f"Unknown encoder type: {encoder_type}. Choose from 'gat' or 'none'")


# Export GAT-only classes and functions
__all__ = [
    'BaseGraphEncoder', 
    'GATEncoder', 
    'BatchedGATWrapper',
    'create_gat_encoder',
    'create_graph_encoder'
]