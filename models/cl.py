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

    def forward(self, views_data_list):
        """
        Args:
            views_data_list (list): A list of tuples. Each tuple contains data for one view.
                                     Example for 2 views:
                                     [
                                         (view1_features_list, view1_adj_mats_list), # Data for view 1
                                         (view2_features_list, view2_adj_mats_list)  # Data for view 2
                                     ]
                                     - viewX_features_list: List of feature tensors [K, Df] for view X, one per WSI in batch slice.
                                     - viewX_adj_mats_list: List of adj matrices [K, K] or None for view X, one per WSI in batch slice.
        Returns:
            tuple: (z_projections, h_bag_embeddings)
                   - z_projections (list): List of projected embeddings [B_slice, projection_dim] for each view.
                   - h_bag_embeddings (list): List of bag embeddings [B_slice, n_features] for each view.
        """
        z_projections = []
        h_bag_embeddings = []

        for view_idx in range(len(views_data_list)):
            current_view_features_list, current_view_adj_mats_list = views_data_list[view_idx]
            
            # self.encoder is the GraphAndMILPipeline
            # It expects: features_batch (list of [K,Df] tensors), adj_mats_batch (list of [K,K] or None tensors)
            # It returns: (bag_embeddings_batch_tensor [B_slice, n_features], 
            #              processed_features_list [list of [K, D_gnn_out] tensors])
            
            # The GraphAndMILPipeline (self.encoder) is already wrapped in DataParallel if multiple GPUs.
            # When cl.CL.forward is called by DataParallel, views_data_list[view_idx] will already be
            # the data for the current GPU's slice of the batch for that view.
            
            bag_embeddings_for_view, _ = self.encoder(
                features_batch=current_view_features_list, 
                adj_mats_batch=current_view_adj_mats_list
            )
            
            projected_output = self.projection_head(bag_embeddings_for_view)
            
            z_projections.append(projected_output)
            h_bag_embeddings.append(bag_embeddings_for_view)
            
        return z_projections, h_bag_embeddings

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