import torch
import torch.nn as nn
from typing import List, Optional, Tuple, Union
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

class GraphAndMILPipeline(nn.Module):
    """
    Coherent pipeline combining Graph processing with MIL aggregation.
    Handles the full SG-MuRCL processing flow.
    """
    
    def __init__(self, 
                 input_dim: int,
                 graph_encoder: Optional[nn.Module] = None,
                 mil_aggregator: Optional[nn.Module] = None):
        super().__init__()
        self.input_dim = input_dim
        self.graph_encoder = graph_encoder
        self.mil_aggregator = mil_aggregator
        
        # Determine feature dimension after graph processing
        if graph_encoder is not None:
            self.post_graph_dim = getattr(graph_encoder, 'output_dim', input_dim)
        else:
            self.post_graph_dim = input_dim
        
        # Determine final output dimension
        if mil_aggregator is not None:
            self.output_dim = getattr(mil_aggregator, 'output_dim', self.post_graph_dim)
        else:
            self.output_dim = self.post_graph_dim

    def forward(self, 
                features_batch: List[torch.Tensor], 
                adj_mats_batch: Optional[List[Optional[torch.Tensor]]] = None,
                **kwargs) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Process batch of WSIs through Graph + MIL pipeline.
        
        Args:
            features_batch: List of [num_patches_i, input_dim] tensors from WSIWithCluster
            adj_mats_batch: List of [num_patches_i, num_patches_i] adjacency matrices from clustering
            **kwargs: Additional arguments (mask, etc.)
            
        Returns:
            bag_embeddings: [batch_size, output_dim] - WSI-level representations
            intermediate_features: List of processed patch features for each WSI
        """
        batch_size = len(features_batch)
        
        # Ensure adj_mats_batch has correct length
        if adj_mats_batch is None:
            adj_mats_batch = [None] * batch_size
        elif len(adj_mats_batch) != batch_size:
            logger.warning(f"adj_mats_batch length {len(adj_mats_batch)} != batch_size {batch_size}")
            # Pad with None values if needed
            adj_mats_batch = adj_mats_batch[:batch_size] + [None] * max(0, batch_size - len(adj_mats_batch))
        
        # Step 1: Graph Processing (GAT)
        if self.graph_encoder is not None:
            try:
                # Use forward method for BatchedGATWrapper
                processed_features = self.graph_encoder(features_batch, adj_mats_batch)
            except Exception as e:
                logger.warning(f"Graph processing failed: {e}, using original features")
                processed_features = features_batch
        else:
            processed_features = features_batch
        
        # Step 2: MIL Aggregation
        bag_embeddings = []
        
        for i, features in enumerate(processed_features):
            if features.numel() == 0:
                # Handle empty WSIs
                empty_embedding = torch.zeros(self.output_dim, device=features.device, dtype=features.dtype)
                bag_embeddings.append(empty_embedding)
                continue
            
            try:
                if self.mil_aggregator is not None:
                    # SmTABMIL expects single WSI format: [num_patches, feat_dim]
                    # It will handle batching internally
                    adj_mat_i = adj_mats_batch[i] if i < len(adj_mats_batch) else None
                    
                    # Check if MIL aggregator accepts adjacency matrices
                    if self._mil_accepts_adj() and adj_mat_i is not None:
                        # Pass adjacency matrix to SmTransformerSmABMIL
                        bag_embedding = self.mil_aggregator(features, adj_mat=adj_mat_i)
                    else:
                        # Standard MIL aggregation (ABMIL)
                        bag_embedding = self.mil_aggregator(features)
                    
                    # Handle different MIL return formats
                    if isinstance(bag_embedding, (list, tuple)):
                        bag_embedding = bag_embedding[0]  # Take first element if multiple outputs
                    
                    # Ensure bag_embedding is 1D for single WSI
                    if bag_embedding.dim() > 1:
                        bag_embedding = bag_embedding.squeeze()
                        if bag_embedding.dim() == 0:  # Handle scalar case
                            bag_embedding = bag_embedding.unsqueeze(0)
                else:
                    # Fallback: mean pooling
                    bag_embedding = torch.mean(features, dim=0)
                
            except Exception as e:
                logger.warning(f"MIL aggregation failed for WSI {i}: {e}")
                # Fallback to mean pooling
                bag_embedding = torch.mean(features, dim=0)
                
                # Match expected output dimension
                if bag_embedding.shape[0] != self.output_dim:
                    if hasattr(self, '_fallback_projection'):
                        bag_embedding = self._fallback_projection(bag_embedding)
                    else:
                        # Create fallback projection on the fly
                        self._fallback_projection = nn.Linear(
                            bag_embedding.shape[0], 
                            self.output_dim
                        ).to(bag_embedding.device)
                        bag_embedding = self._fallback_projection(bag_embedding)
            
            bag_embeddings.append(bag_embedding)
        
        # Stack bag embeddings
        if bag_embeddings:
            batched_embeddings = torch.stack(bag_embeddings, dim=0)  # [batch_size, output_dim]
        else:
            # Handle empty batch case
            device = features_batch[0].device if features_batch else torch.device('cpu')
            batched_embeddings = torch.empty(0, self.output_dim, device=device)
        
        return batched_embeddings, processed_features
    
    def _mil_accepts_adj(self) -> bool:
        """Check if MIL aggregator can accept adjacency matrices."""
        if self.mil_aggregator is None:
            return False
        
        # Check for SmTransformerSmABMIL or similar models that use adjacency matrices
        class_name = self.mil_aggregator.__class__.__name__
        return any(keyword in class_name for keyword in ['SmTransformer', 'Sm', 'Graph'])
    
    def get_attention_weights(self, 
                            features_batch: List[torch.Tensor], 
                            adj_mats_batch: Optional[List[Optional[torch.Tensor]]] = None) -> List[torch.Tensor]:
        """Extract attention weights from MIL aggregator for visualization."""
        if self.mil_aggregator is None:
            return [None] * len(features_batch)
        
        # Process features through graph encoder first
        if self.graph_encoder is not None:
            try:
                processed_features = self.graph_encoder(features_batch, adj_mats_batch or [None] * len(features_batch))
            except Exception as e:
                logger.warning(f"Graph processing failed in attention extraction: {e}")
                processed_features = features_batch
        else:
            processed_features = features_batch
        
        attention_weights = []
        for i, features in enumerate(processed_features):
            if features.numel() == 0:
                attention_weights.append(None)
                continue
            
            try:
                # Different methods to extract attention weights
                adj_mat_i = adj_mats_batch[i] if adj_mats_batch and i < len(adj_mats_batch) else None
                
                if hasattr(self.mil_aggregator, 'get_attention_weights'):
                    # Method 1: Direct attention weights method
                    weights = self.mil_aggregator.get_attention_weights(features, adj_mat=adj_mat_i)
                    attention_weights.append(weights)
                elif hasattr(self.mil_aggregator, 'attention_pool'):
                    # Method 2: SmTABMIL-style attention pooling
                    try:
                        # Process through embedding and transformer first
                        h = self.mil_aggregator.emb_layer(features)
                        h = self.mil_aggregator.dropout_layer(h)
                        h = h.unsqueeze(0)  # Add batch dim
                        h_transformed, _ = self.mil_aggregator.transformer_encoder(h, adj_mat=adj_mat_i)
                        
                        # Get attention weights from attention pooling
                        _, attention = self.mil_aggregator.attention_pool(h_transformed, adj_mat=adj_mat_i)
                        attention_weights.append(attention.squeeze(0) if attention is not None else None)
                    except Exception as e:
                        logger.debug(f"Failed to extract SmTABMIL attention: {e}")
                        attention_weights.append(None)
                else:
                    # Method 3: No attention available
                    attention_weights.append(None)
                    
            except Exception as e:
                logger.warning(f"Failed to extract attention weights for WSI {i}: {e}")
                attention_weights.append(None)
        
        return attention_weights


# Factory function for easy pipeline creation
def create_pipeline(input_dim: int, 
                   graph_encoder_type: str = "none",
                   mil_aggregator_type: str = "abmil",
                   **kwargs) -> GraphAndMILPipeline:
    """
    Factory function to create GraphAndMILPipeline with specified components.
    
    Args:
        input_dim (int): Input feature dimension
        graph_encoder_type (str): Type of graph encoder ('gat', 'none')
        mil_aggregator_type (str): Type of MIL aggregator ('smtabmil', 'abmil')
        **kwargs: Additional arguments for component creation
        
    Returns:
        GraphAndMILPipeline: Configured pipeline
    """
    # Create graph encoder
    graph_encoder = None
    if graph_encoder_type.lower() == "gat":
        from .graph_encoders import create_gat_encoder
        graph_encoder = create_gat_encoder(
            input_dim=input_dim,
            output_dim=input_dim,  # Keep same dimension by default
            **kwargs.get('gat_args', {})
        )
    
    # Create MIL aggregator
    mil_aggregator = None
    if mil_aggregator_type.lower() == "smtabmil":
        from .smtabmil import create_smtabmil
        mil_aggregator = create_smtabmil(
            feature_dim=input_dim,
            **kwargs.get('smtabmil_args', {})
        )
    elif mil_aggregator_type.lower() == "abmil":
        from .abmil import ABMIL
        mil_aggregator = ABMIL(
            dim_in=input_dim,
            **kwargs.get('abmil_args', {})
        )
    
    return GraphAndMILPipeline(
        input_dim=input_dim,
        graph_encoder=graph_encoder,
        mil_aggregator=mil_aggregator
    )


# Export classes and functions
__all__ = [
    'GraphAndMILPipeline',
    'create_pipeline'
]