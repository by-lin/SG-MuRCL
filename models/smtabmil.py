import torch
import torch.nn as nn
from .modules.MILTransformer import TransformerEncoder
from .modules.MILAttentionPool import MILAttentionPool

class SMTABMIL(nn.Module):
    """
    SmTransformer + SmABMIL implementation matching SmMIL's DeepMILAttModel pattern 
    for 'sm_transformer_abmil'. This is equivalent to DeepMILAttModel with both 
    transformer_encoder_kwargs and pool_kwargs containing SmMix parameters.
    """
    def __init__(self, input_dim, emb_dim=512, 
                 transformer_encoder_kwargs=None, 
                 pool_kwargs=None, 
                 ce_criterion=None,
                 **kwargs):
        """
        Args:
            input_dim: Input feature dimension
            emb_dim: Embedding dimension
            transformer_encoder_kwargs: Dict with transformer parameters:
                - att_dim: Transformer attention dimension
                - num_heads: Number of attention heads
                - num_layers: Number of transformer layers
                - use_ff: Use feed-forward layers
                - dropout: Dropout rate
                - use_sm: Enable SmMix in transformer
                - sm_alpha: SmMix alpha parameter
                - sm_mode: SmMix mode ('approx' or 'exact')
                - sm_steps: SmMix steps
            pool_kwargs: Dict with pooling parameters:
                - att_dim: Pooling attention dimension
                - sm_alpha: SmMix alpha for pooling
                - sm_mode: SmMix mode for pooling
                - sm_where: Where to apply SmMix ('early', 'late', 'both')
                - sm_steps: SmMix steps for pooling
                - sm_spectral_norm: Use spectral normalization
            ce_criterion: Loss criterion (for compatibility, not used in forward)
        """
        super().__init__()
        
        # Set defaults matching SmMIL's sm_transformer_abmil
        if transformer_encoder_kwargs is None:
            transformer_encoder_kwargs = {
                'att_dim': 128,
                'num_heads': 8,
                'num_layers': 2,
                'use_ff': True,
                'dropout': 0.1,
                'use_sm': True,
                'sm_alpha': 0.5,
                'sm_mode': 'approx',
                'sm_steps': 10,
            }
        if pool_kwargs is None:
            pool_kwargs = {
                'att_dim': 128,
                'sm_alpha': 0.5,
                'sm_mode': 'approx',
                'sm_where': 'early',
                'sm_steps': 10,
                'sm_spectral_norm': True,
            }
            
        self.input_dim = input_dim
        self.emb_dim = emb_dim
        self.output_dim = emb_dim
        self.ce_criterion = ce_criterion

        # Feature extraction layer (matching DeepMILAttModel)
        self.feat_ext = nn.Sequential(
            nn.Linear(input_dim, emb_dim),
            nn.GELU()
        )
        
        # SmTransformer Encoder (matching DeepMILAttModel pattern)
        self.transformer_encoder = TransformerEncoder(
            in_dim=emb_dim,
            **transformer_encoder_kwargs
        )
        
        # SmMIL Attention Pooling (using SmMIL's exact implementation)
        self.pool = MILAttentionPool(
            in_dim=emb_dim,
            **pool_kwargs
        )
        
        # Classifier head (matching DeepMILAttModel)
        self.classifier = nn.Linear(emb_dim, 1)
        
    def forward(self, features_batch, adj_mat=None, mask=None, return_att=False, return_loss=False, **kwargs):
        """
        Forward pass with proper return values for MuRCL and heatmap generation.
        """
        # Feature extraction
        X = self.feat_ext(features_batch)  # [B, N, emb_dim]
        
        # SmTransformer processing
        if self.transformer_encoder is not None:
            X_transformed = self.transformer_encoder(X, adj_mat=adj_mat, mask=mask)
        else:
            X_transformed = X
        
        # SmMIL attention pooling
        if return_att:
            # Get attention weights for heatmap generation
            bag_embedding, attention_weights = self.pool(X_transformed, adj_mat=adj_mat, mask=mask, return_att=True)
            return bag_embedding, attention_weights
        else:
            # Get bag embedding from SmMIL pooling
            bag_embedding = self.pool(X_transformed, adj_mat=adj_mat, mask=mask, return_att=False)
            
            # For MuRCL: return different ppo_state
            # PPO state should be the transformer output (for policy learning)
            ppo_state = X_transformed.mean(dim=1)  # [B, emb_dim] - mean pooling of transformer output
            return bag_embedding, ppo_state

def create_smtabmil(input_dim, **kwargs):
    """Factory function matching SmMIL pattern."""
    return SMTABMIL(input_dim=input_dim, **kwargs)