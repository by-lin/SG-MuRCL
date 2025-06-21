import torch
import torch.nn as nn
import logging

logger = logging.getLogger(__name__)

class CL(nn.Module):
    """Contrastive Learning wrapper following original pattern."""
    
    def __init__(self, encoder, projection_dim, n_features):
        super(CL, self).__init__()
        self.encoder = encoder
        self.projection_dim = projection_dim
        self.n_features = n_features
        
        # Add projector like original (but actually use it)
        self.projector = nn.Sequential(
            nn.Linear(n_features, n_features),
            nn.ReLU(),
            nn.Linear(n_features, projection_dim)
        )

    def forward(self, x_views):
        """
        Forward pass following original pattern.
        
        Args:
            x_views: List of input views (either tensors or tuples)
            
        Returns:
            h_views: List of encoder outputs (for FC layer) 
            states: List of detached encoder outputs (for PPO)
        """
        assert isinstance(x_views, list), "x_views must be a list"
        
        # Forward through encoder - handle both simple and complex inputs
        h_views = []
        for x in x_views:
            if isinstance(x, tuple):
                # For graph/advanced models: (features, adj_mat, masks)
                features, adj_mat, masks = x
                encoder_output = self.encoder(features, adj_mat=adj_mat, masks=masks)
            else:
                # For simple models: just features
                encoder_output = self.encoder(x)
            
            # Handle encoder output format
            if isinstance(encoder_output, tuple):
                # If encoder returns tuple, take first element like original
                h = encoder_output[0]
            else:
                # If encoder returns single tensor
                h = encoder_output
            
            h_views.append(h)
        
        # Return like original: (outputs, detached_outputs)
        return h_views, [h.detach() for h in h_views]