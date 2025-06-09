import torch
import torch.nn as nn
from .modules.MILTransformer import TransformerEncoder
from .modules.MILAttentionPool import MILAttentionPool

class SmTransformerSmABMIL(nn.Module):
    """
    SmTransformerSmABMIL: Transformer with Sm + Attention-based MIL with Sm.
    Adapts SmMIL's transformer and attention pooling for MuRCL.
    """
    def __init__(self, dim_in, L=512, D=128, dropout=0.0,
                 # Transformer args
                 transf_num_heads=4, transf_num_layers=2, transf_use_ff=True,
                 transf_dropout=0.1, use_sm_transformer=True,
                 # SmMIL args for both Transformer and Attention Pool
                 sm_alpha=0.5, sm_mode='approx', sm_where='early', # sm_where for attention pool
                 sm_steps=10, sm_spectral_norm=True, **kwargs): # Added **kwargs to absorb unused num_classes
        super(SmTransformerSmABMIL, self).__init__()
        
        self.dim_in = dim_in
        self.L = L # Dimension of the embedding space / bag representation
        self.D = D # Dimension for attention mechanism

        # Embedding layer
        self.emb_layer = nn.Linear(dim_in, L)
        self.dropout_layer = nn.Dropout(dropout)
        
        # Transformer Encoder with optional Sm operator
        self.transformer_encoder = TransformerEncoder(
            in_dim=L,
            att_dim=D, # Internal attention dimension for transformer
            num_heads=transf_num_heads,
            num_layers=transf_num_layers,
            use_ff=transf_use_ff,
            dropout=transf_dropout,
            use_sm=use_sm_transformer, # Whether Transformer uses Sm
            sm_alpha=sm_alpha,
            sm_mode=sm_mode,
            sm_steps=sm_steps
        )
        
        # SmMIL Attention Pooling with Sm operator
        self.attention_pool = MILAttentionPool(
            att_dim=D,
            in_dim=L, # Input from Transformer is also L
            sm_alpha=sm_alpha,
            sm_mode=sm_mode,
            sm_where=sm_where, # Controls Sm in attention pooling
            sm_steps=sm_steps,
            spectral_norm=sm_spectral_norm
        )
        
        # Store output dimension for pipeline compatibility
        self.output_dim = L
        
    def forward(self, x, adj_mat=None, mask=None):
        """
        Forward pass compatible with MuRCL pipeline.
        
        Args:
            x (torch.Tensor): Input patch features 
                             - Single WSI: [num_patches, dim_in]
                             - Batched: [batch_size, num_patches, dim_in] 
            adj_mat (torch.Tensor, optional): Adjacency matrix for Sm operator
            mask (torch.Tensor, optional): Attention mask
            
        Returns:
            torch.Tensor: Bag representation [L] for single WSI or [batch_size, L] for batch
        """
        # Determine if input is batched or single WSI
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
            adj_mat (torch.Tensor, optional): [num_patches, num_patches]
            mask (torch.Tensor, optional): [num_patches]
            
        Returns:
            torch.Tensor: Bag representation [L]
        """
        # Add batch dimension for transformer processing
        x = x.unsqueeze(0)  # [1, num_patches, dim_in]
        
        if adj_mat is not None:
            adj_mat = adj_mat.unsqueeze(0)  # [1, num_patches, num_patches]
        
        if mask is not None:
            mask = mask.unsqueeze(0)  # [1, num_patches]
        
        # Process through the network
        bag_features = self._forward_batch(x, adj_mat, mask)  # [1, L]
        
        # Remove batch dimension
        return bag_features.squeeze(0)  # [L]
    
    def _forward_batch(self, x, adj_mat=None, mask=None):
        """
        Forward pass for batched input.
        
        Args:
            x (torch.Tensor): [batch_size, num_patches, dim_in]
            adj_mat (torch.Tensor, optional): [batch_size, num_patches, num_patches]
            mask (torch.Tensor, optional): [batch_size, num_patches]
            
        Returns:
            torch.Tensor: Bag representations [batch_size, L]
        """
        # Embedding
        h = self.emb_layer(x)  # [batch_size, num_patches, L]
        h = self.dropout_layer(h)
        
        # Transformer encoding with optional Sm operator
        # TransformerEncoder returns (transformed_features, list_of_attention_weights_from_each_transformer_layer)
        h_transformed, _ = self.transformer_encoder(h, adj_mat=adj_mat, mask=mask)
        # h_transformed: [batch_size, num_patches, L]
        
        # Attention pooling with Sm operator on transformer's output
        pooled_features, attention_weights = self.attention_pool(h_transformed, adj_mat=adj_mat, mask=mask)
        # pooled_features: [batch_size, L]
        
        return pooled_features


def build_smtransformer_smabmil(dim_feat, L=512, D=128, dropout=0.0,
                                transf_num_heads=4, transf_num_layers=2, transf_use_ff=True,
                                transf_dropout=0.1, use_sm_transformer=True,
                                sm_alpha=0.5, sm_mode='approx', sm_where='early',
                                sm_steps=10, sm_spectral_norm=True): # num_classes for signature compatibility
    """
    Builder function for SmTransformerSmABMIL.
    
    Args:
        dim_feat (int): Input feature dimension
        L (int): Embedding dimension / bag representation dimension
        D (int): Attention dimension
        num_classes (int): For compatibility (not used in MuRCL contrastive learning)
        **kwargs: Additional arguments
        
    Returns:
        SmTransformerSmABMIL: Configured model ready for MuRCL pipeline
    """
    model = SmTransformerSmABMIL(
        dim_in=dim_feat,
        L=L,
        D=D,
        dropout=dropout,
        transf_num_heads=transf_num_heads,
        transf_num_layers=transf_num_layers,
        transf_use_ff=transf_use_ff,
        transf_dropout=transf_dropout,
        use_sm_transformer=use_sm_transformer,
        sm_alpha=sm_alpha,
        sm_mode=sm_mode,
        sm_where=sm_where,
        sm_steps=sm_steps,
        sm_spectral_norm=sm_spectral_norm
    )
    return model


# Alias for easier integration with existing MuRCL code
class SMTABMIL(SmTransformerSmABMIL):
    """Alias for easier integration with existing MuRCL code."""
    def __init__(self, dim_in, L=512, D=128, dropout=0.0, **kwargs):
        super().__init__(dim_in=dim_in, L=L, D=D, dropout=dropout, **kwargs)


def create_smtabmil(feature_dim, **kwargs):
    """
    Factory function to create SMTABMIL model for MuRCL.
    
    Args:
        feature_dim (int): Input feature dimension (e.g., 1024)
        **kwargs: Additional configuration
        
    Returns:
        SMTABMIL: Model ready for MuRCL training
    """
    return SMTABMIL(dim_in=feature_dim, **kwargs)


# Export main classes and functions
__all__ = [
    'SmTransformerSmABMIL', 
    'SMTABMIL', 
    'build_smtransformer_smabmil', 
    'create_smtabmil'
]