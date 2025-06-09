import math
import torch
import torch.nn.functional as F
from torch import nn


class ABMIL(nn.Module):
    def __init__(self, dim_in, L=512, D=128, K=1, dropout=0.):
        super(ABMIL, self).__init__()
        self.dim_in = dim_in
        self.L = L
        self.D = D
        self.K = K
        
        # Store output dimension for pipeline compatibility
        self.output_dim = L

        # Feature encoder
        self.encoder = nn.Sequential(
            nn.Linear(dim_in, L),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(L, L),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(L, L),
            nn.ReLU(),
        )

        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(L, D),
            nn.Tanh(),
            nn.Linear(D, K)
        )

        # Final decoder
        self.decoder = nn.Sequential(
            nn.Linear(L, L),
            nn.ReLU(),
        )

    def forward(self, x, adj_mat=None, mask=None):
        """
        Forward pass compatible with MuRCL pipeline.
        
        Args:
            x (torch.Tensor): Input patch features 
                             - Single WSI: [num_patches, dim_in]
                             - Batched: [batch_size, num_patches, dim_in] 
            adj_mat: Ignored by ABMIL (kept for interface compatibility)
            mask: Ignored by ABMIL (kept for interface compatibility)
            
        Returns:
            torch.Tensor: Bag representation [L] for single WSI or [batch_size, L] for batch
        """
        # Determine if input is batched or single WSI (SAME AS SMTABMIL)
        if len(x.shape) == 2:  # Single WSI: [num_patches, dim_in]
            return self._forward_single(x, adj_mat, mask)
        elif len(x.shape) == 3:  # Batched: [batch_size, num_patches, dim_in]
            return self._forward_batch(x, adj_mat, mask)
        else:
            raise ValueError(f"Expected input shape [num_patches, dim_in] or [batch_size, num_patches, dim_in], got {x.shape}")
    
    def _forward_single(self, x, adj_mat=None, mask=None):
        """
        Forward pass for single WSI (used by MuRCL pipeline).
        
        Args:
            x (torch.Tensor): [num_patches, dim_in]
            adj_mat: Ignored by ABMIL
            mask: Ignored by ABMIL
            
        Returns:
            torch.Tensor: Bag representation [L]
        """
        # Add batch dimension for processing (SAME PATTERN AS SMTABMIL)
        x = x.unsqueeze(0)  # [1, num_patches, dim_in]
        
        # Process through the network
        bag_features = self._forward_batch(x, adj_mat, mask)  # [1, L]
        
        # Remove batch dimension
        return bag_features.squeeze(0)  # [L]
    
    def _forward_batch(self, x, adj_mat=None, mask=None):
        """
        Forward pass for batched input.
        
        Args:
            x (torch.Tensor): [batch_size, num_patches, dim_in]
            adj_mat: Ignored by ABMIL
            mask: Ignored by ABMIL
            
        Returns:
            torch.Tensor: Bag representations [batch_size, L]
        """
        batch_size, num_patches, dim_in = x.shape
        
        # Handle empty input
        if num_patches == 0:
            return torch.zeros(batch_size, self.L, device=x.device, dtype=x.dtype)
        
        # Reshape for batch processing
        x_flat = x.view(-1, dim_in)  # [batch_size * num_patches, dim_in]
        
        # Feature encoding
        H_flat = self.encoder(x_flat)  # [batch_size * num_patches, L]
        H = H_flat.view(batch_size, num_patches, self.L)  # [batch_size, num_patches, L]

        # Attention computation
        A_raw_flat = self.attention(H_flat)  # [batch_size * num_patches, K]
        A_raw = A_raw_flat.view(batch_size, num_patches, self.K)  # [batch_size, num_patches, K]
        
        # Transpose for softmax over patches
        A_transposed = A_raw.transpose(1, 2)  # [batch_size, K, num_patches]
        A_softmax = F.softmax(A_transposed, dim=2)  # [batch_size, K, num_patches]
        
        # Attention pooling
        if self.K == 1:
            # Single attention head
            A_weights = A_softmax.squeeze(1)  # [batch_size, num_patches]
            bag_representation = torch.bmm(A_weights.unsqueeze(1), H).squeeze(1)  # [batch_size, L]
        else:
            # Multiple attention heads - average them
            attention_weighted = torch.bmm(A_softmax, H)  # [batch_size, K, L]
            bag_representation = torch.mean(attention_weighted, dim=1)  # [batch_size, L]
        
        # Final processing
        output = self.decoder(bag_representation)  # [batch_size, L]
        
        return output

    def get_attention_weights(self, x):
        """
        Extract attention weights for visualization.
        
        Args:
            x (torch.Tensor): [num_patches, dim_in] - single WSI only
            
        Returns:
            torch.Tensor: [num_patches] - attention weights
        """
        if x.numel() == 0:
            return None
            
        try:
            # Add batch dimension
            x_batch = x.unsqueeze(0)  # [1, num_patches, dim_in]
            
            # Get features
            x_flat = x_batch.view(-1, self.dim_in)  # [num_patches, dim_in]
            H_flat = self.encoder(x_flat)  # [num_patches, L]
            H = H_flat.view(1, x.shape[0], self.L)  # [1, num_patches, L]
            
            # Get attention
            A_raw_flat = self.attention(H_flat)  # [num_patches, K]
            A_raw = A_raw_flat.view(1, x.shape[0], self.K)  # [1, num_patches, K]
            A_transposed = A_raw.transpose(1, 2)  # [1, K, num_patches]
            A_softmax = F.softmax(A_transposed, dim=2)  # [1, K, num_patches]
            
            # Return attention from first head
            return A_softmax[0, 0, :]  # [num_patches]
        except Exception:
            return None


# Factory function matching SmTABMIL pattern
def build_abmil(dim_feat, L=512, D=128, dropout=0.0, num_classes=1, **kwargs):
    """
    Builder function for ABMIL (matching SmTABMIL signature).
    
    Args:
        dim_feat (int): Input feature dimension
        L (int): Embedding dimension / bag representation dimension
        D (int): Attention dimension
        num_classes (int): For compatibility (not used in MuRCL contrastive learning)
        **kwargs: Additional arguments
        
    Returns:
        ABMIL: Configured model ready for MuRCL pipeline
    """
    model = ABMIL(
        dim_in=dim_feat,
        L=L,
        D=D,
        dropout=dropout,
        **kwargs
    )
    return model


def create_abmil(feature_dim, **kwargs):
    """
    Factory function to create ABMIL model for MuRCL.
    
    Args:
        feature_dim (int): Input feature dimension (e.g., 1024)
        **kwargs: Additional configuration
        
    Returns:
        ABMIL: Model ready for MuRCL training
    """
    return ABMIL(dim_in=feature_dim, **kwargs)


# Export classes and functions
__all__ = ['ABMIL', 'build_abmil', 'create_abmil']