import torch
import torch.nn as nn
import logging

logger = logging.getLogger(__name__)

class CL(nn.Module):
    def __init__(self, encoder, projection_dim, n_features):
        super(CL, self).__init__()

        self.encoder = encoder  # This is now GraphAndMILPipeline
        self.n_features = n_features  # Output dimension of the pipeline
        self.projection_dim = projection_dim

        # Projection head for contrastive learning
        self.projection_head = nn.Sequential(
            nn.Linear(self.n_features, self.n_features, bias=False),
            nn.ReLU(),
            nn.Linear(self.n_features, projection_dim, bias=False)
        )

    def forward(self, x_views_list, adj_mats=None):
        """
        Forward pass for contrastive learning with GraphAndMILPipeline.
        
        Args:
            x_views_list (list): List of two views, each containing a list of WSI feature tensors
                                Format: [[view1_wsi1, view1_wsi2, ...], [view2_wsi1, view2_wsi2, ...]]
                                Each WSI tensor: [num_patches_i, feature_dim]
            adj_mats (list, optional): List of two views of adjacency matrices
                                      Format: [[view1_adj1, view1_adj2, ...], [view2_adj1, view2_adj2, ...]]
                                      Each adj matrix: [num_patches_i, num_patches_i]
        
        Returns:
            tuple: (projections, bag_embeddings, states)
                   projections: List of [batch_size, projection_dim] tensors for each view
                   bag_embeddings: List of [batch_size, n_features] tensors for each view  
                   states: List of processed features for PPO (for each view)
        """
        if len(x_views_list) != 2:
            raise ValueError(f"Expected 2 views for contrastive learning, got {len(x_views_list)}")
        
        projections = []
        bag_embeddings = []
        all_states = []

        for view_idx, x_view_list in enumerate(x_views_list):
            # Get corresponding adjacency matrices for this view
            adj_mats_view = None
            if adj_mats is not None and view_idx < len(adj_mats):
                adj_mats_view = adj_mats[view_idx]
            
            try:
                # Forward through the pipeline (GraphAndMILPipeline)
                # Pipeline expects: features_batch (List[Tensor]), adj_mats_batch (List[Tensor])
                # Pipeline returns: (bag_embeddings [batch_size, output_dim], processed_features [List])
                bag_embeddings_view, states_view = self.encoder(x_view_list, adj_mats_view)
                
                # Project bag embeddings for contrastive learning
                projections_view = self.projection_head(bag_embeddings_view)
                
                projections.append(projections_view)
                bag_embeddings.append(bag_embeddings_view)
                all_states.append(states_view)
                
            except Exception as e:
                logger.error(f"CL forward failed for view {view_idx}: {e}")
                # Fallback: create zero tensors
                batch_size = len(x_view_list)
                device = x_view_list[0].device if x_view_list else torch.device('cpu')
                
                zero_embeddings = torch.zeros(batch_size, self.n_features, device=device)
                zero_projections = torch.zeros(batch_size, self.projection_dim, device=device)
                
                projections.append(zero_projections)
                bag_embeddings.append(zero_embeddings)
                all_states.append(x_view_list)  # Use original features as fallback
        
        return projections, bag_embeddings, all_states

    def encode(self, x_list, adj_mats=None):
        """
        Encode WSIs without projection (for evaluation/inference).
        
        Args:
            x_list (list): List of WSI feature tensors [num_patches_i, feature_dim]
            adj_mats (list, optional): List of adjacency matrices [num_patches_i, num_patches_i]
            
        Returns:
            torch.Tensor: Bag embeddings [batch_size, n_features]
        """
        try:
            bag_embeddings, _ = self.encoder(x_list, adj_mats)
            return bag_embeddings
        except Exception as e:
            logger.error(f"CL encode failed: {e}")
            # Fallback: zero embeddings
            batch_size = len(x_list)
            device = x_list[0].device if x_list else torch.device('cpu')
            return torch.zeros(batch_size, self.n_features, device=device)

    def get_attention_weights(self, x_list, adj_mats=None):
        """
        Extract attention weights for visualization.
        
        Args:
            x_list (list): List of WSI feature tensors
            adj_mats (list, optional): List of adjacency matrices
            
        Returns:
            list: Attention weights for each WSI (or None if not available)
        """
        if hasattr(self.encoder, 'get_attention_weights'):
            return self.encoder.get_attention_weights(x_list, adj_mats)
        else:
            logger.warning("Encoder does not support attention weight extraction")
            return [None] * len(x_list)

    def get_feature_dim(self):
        """Get the feature dimension of the encoder output."""
        return self.n_features

    def get_projection_dim(self):
        """Get the projection dimension for contrastive learning."""
        return self.projection_dim


# Factory function for easier creation
def create_cl_model(encoder, projection_dim=128):
    """
    Create CL model with automatic feature dimension detection.
    
    Args:
        encoder: The encoder (should be GraphAndMILPipeline)
        projection_dim (int): Dimension for contrastive projection
        
    Returns:
        CL: Configured contrastive learning model
    """
    # Get feature dimension from encoder
    if hasattr(encoder, 'output_dim'):
        n_features = encoder.output_dim
    else:
        # Fallback: assume common dimension
        n_features = 512
        logger.warning(f"Could not detect encoder output dimension, using {n_features}")
    
    return CL(
        encoder=encoder,
        projection_dim=projection_dim,
        n_features=n_features
    )


# Export classes and functions
__all__ = ['CL', 'create_cl_model']