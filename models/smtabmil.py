import torch
import torch.nn as nn
from .modules.MILTransformer import TransformerEncoder
from .modules.MILAttentionPool import MILAttentionPool

class SMTABMIL(nn.Module):
    """
    Vectorized SmTransformer + Attention-based MIL.
    """
    def __init__(self, dim_in, L=512, D=128, dropout=0.0,
                 transf_num_heads=4, transf_num_layers=2, transf_use_ff=True,
                 transf_dropout=0.1, use_sm_transformer=True,
                 sm_alpha=0.5, sm_mode='approx', sm_where='early',
                 sm_steps=10, sm_spectral_norm=True, **kwargs):
        super().__init__()
        self.L = L
        self.output_dim = L

        self.emb_layer = nn.Linear(dim_in, L)
        self.dropout_layer = nn.Dropout(dropout)
        
        self.transformer_encoder = TransformerEncoder(
            in_dim=L,
            att_dim=D,
            num_heads=transf_num_heads,
            num_layers=transf_num_layers,
            use_ff=transf_use_ff,
            dropout=transf_dropout,
            use_sm=use_sm_transformer,
            sm_alpha=sm_alpha,
            sm_mode=sm_mode,
            sm_steps=sm_steps
        )
        
        self.attention_pool = MILAttentionPool(
            att_dim=D,
            in_dim=L,
            sm_alpha=sm_alpha,
            sm_mode=sm_mode,
            sm_where=sm_where,
            sm_steps=sm_steps,
            spectral_norm=sm_spectral_norm
        )
        
    def forward(self, 
                features_batch: torch.Tensor, 
                adj_mat: torch.Tensor = None, 
                mask: torch.Tensor = None,
                **kwargs):
        """
        Fully-batched forward pass for SMTABMIL.

        Args:
            features_batch (torch.Tensor): [B, N, D_in]
            adj_mat (torch.Tensor, optional): [B, N, N]
            mask (torch.Tensor, optional): [B, N]

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: 
                - Final bag embeddings, shape [B, L].
                - PPO states (same as bag embeddings), shape [B, L].
        """
        h = self.emb_layer(features_batch)  # [B, N, L]
        h = self.dropout_layer(h)
        
        h_transformed, _ = self.transformer_encoder(h, adj_mat=adj_mat, mask=mask)
        bag_embedding, _ = self.attention_pool(h_transformed, adj_mat=adj_mat, mask=mask)
        ppo_state = bag_embedding
        return bag_embedding, ppo_state

def create_smtabmil(feature_dim, **kwargs):
    """Factory function to create SMTABMIL model."""
    return SMTABMIL(dim_in=feature_dim, **kwargs)