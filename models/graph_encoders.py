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
        super().__init__(input_dim, output_dim)
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.heads = heads
        self.dropout = dropout
        self.concat_heads = concat_heads  # Whether to concatenate or average heads in final layer

        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        
        # Input projection to match GAT expected input
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        # GAT layers
        for i in range(num_layers):
            if i == 0:
                in_channels = hidden_dim
            else:
                # Previous layer output: hidden_dim * heads (concatenated)
                in_channels = hidden_dim * heads
            
            if i == num_layers - 1:
                # Final layer: either concatenate heads or average them
                out_channels = output_dim // heads if concat_heads else output_dim
                concat = concat_heads
            else:
                out_channels = hidden_dim
                concat = True  # Always concatenate in intermediate layers
            
            self.convs.append(GATConv(in_channels, out_channels, heads=heads, dropout=dropout, concat=concat))
            
            # Batch norm for the actual output dimension
            bn_dim = out_channels * heads if concat else out_channels
            self.batch_norms.append(nn.BatchNorm1d(bn_dim))

        # Update output dimension to reflect actual output
        if concat_heads and num_layers > 0:
            self.actual_output_dim = output_dim * heads
        else:
            self.actual_output_dim = output_dim
        
        # Override the output_dim property to match actual output
        self.output_dim = self.actual_output_dim

    def forward(self, x, edge_index, batch=None):
        """
        Forward pass for single graph.
        
        Args:
            x (torch.Tensor): Node features [num_nodes, input_dim]
            edge_index (torch.Tensor): Edge indices [2, num_edges]
            batch (torch.Tensor, optional): Batch vector for multiple graphs
            
        Returns:
            torch.Tensor: Transformed node features [num_nodes, output_dim]
        """
        # Input projection
        x = self.input_proj(x)  # [num_nodes, hidden_dim]
        
        # Apply GAT layers
        for i, (conv, bn) in enumerate(zip(self.convs, self.batch_norms)):
            x = conv(x, edge_index)  # GAT convolution
            
            # Apply batch norm (handle single node case)
            if x.size(0) > 1:
                x = bn(x)
            
            # Apply activation (except last layer)
            if i < len(self.convs) - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        
        return x  # [num_nodes, output_dim]

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
    """Wrapper to handle batched processing for MuRCL pipeline."""
    
    def __init__(self, gat_encoder):
        super().__init__()
        self.gat_encoder = gat_encoder
        self.input_dim = gat_encoder.input_dim
        self.output_dim = gat_encoder.output_dim
        
        # Add dimension matching layer for fallback cases
        if gat_encoder.input_dim != gat_encoder.output_dim:
            self.dim_matching_layer = nn.Linear(gat_encoder.input_dim, gat_encoder.output_dim)
        else:
            self.dim_matching_layer = None
    
    def forward_single(self, features, adj_mat):
        """
        Process single graph (for pipeline compatibility).
        
        Args:
            features (torch.Tensor): [num_nodes, input_dim]
            adj_mat (torch.Tensor): [num_nodes, num_nodes] adjacency matrix
            
        Returns:
            torch.Tensor: [num_nodes, output_dim] processed features
        """
        if features.numel() == 0:
            return torch.zeros(0, self.output_dim, device=features.device, dtype=features.dtype)
        
        try:
            if adj_mat is not None and adj_mat.numel() > 0:
                edge_index = self.gat_encoder.adj_mat_to_edge_index(adj_mat)
                
                if edge_index.numel() > 0:
                    processed = self.gat_encoder(features, edge_index)
                else:
                    logger.debug("Empty edge index, returning dimension-matched features")
                    processed = self._match_dimensions(features)
            else:
                logger.debug("No adjacency matrix, returning dimension-matched features")
                processed = self._match_dimensions(features)
                    
        except Exception as e:
            logger.warning(f"GAT processing failed: {e}, falling back to dimension-matched features")
            processed = self._match_dimensions(features)
        
        return processed
    
    def _match_dimensions(self, features):
        """Apply dimension matching if needed."""
        if self.dim_matching_layer is not None:
            return self.dim_matching_layer(features)
        else:
            return features
    
    def forward(self, features_list, adj_mats_list):
        """
        Process list of graphs (for MuRCL pipeline compatibility).
        
        Args:
            features_list (List[torch.Tensor]): List of [num_nodes_i, input_dim] tensors
            adj_mats_list (List[torch.Tensor]): List of [num_nodes_i, num_nodes_i] adjacency matrices
            
        Returns:
            List[torch.Tensor]: List of [num_nodes_i, output_dim] processed features
        """
        if len(features_list) != len(adj_mats_list):
            raise ValueError(f"Features list length {len(features_list)} != adj mats list length {len(adj_mats_list)}")
        
        processed_features = []
        
        for features, adj_mat in zip(features_list, adj_mats_list):
            processed = self.forward_single(features, adj_mat)
            processed_features.append(processed)
        
        return processed_features


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