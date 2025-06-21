import torch
import torch.nn as nn

class ABMIL(nn.Module):
    """
    ABMIL implementation matching SmMIL's DeepMILAttModel pattern for 'abmil'.
    This is equivalent to DeepMILAttModel with no transformer_encoder_kwargs.
    """
    def __init__(self, input_dim, emb_dim=512, pool_kwargs=None, ce_criterion=None, **kwargs):
        """
        Args:
            input_dim: Input feature dimension
            emb_dim: Embedding dimension
            pool_kwargs: Pooling parameters dict containing:
                - att_dim: Attention dimension
                - K: Number of attention heads
            ce_criterion: Loss criterion (for compatibility, not used in forward)
        """
        super(ABMIL, self).__init__()
        
        if pool_kwargs is None:
            pool_kwargs = {'att_dim': 128, 'K': 1}
            
        self.input_dim = input_dim
        self.emb_dim = emb_dim
        self.output_dim = emb_dim
        self.ce_criterion = ce_criterion
        self.K = pool_kwargs.get('K', 1)

        # Feature extraction layer (matching DeepMILAttModel)
        self.feat_ext = nn.Sequential(
            nn.Linear(input_dim, emb_dim),
            nn.GELU()
        )
        
        # No transformer encoder for basic ABMIL
        self.transformer_encoder = None
        
        # Attention layers for ABMIL (this was missing!)
        self.attention_V = nn.Sequential(
            nn.Linear(emb_dim, pool_kwargs['att_dim']),
            nn.Tanh()
        )
        
        self.attention_U = nn.Sequential(
            nn.Linear(emb_dim, pool_kwargs['att_dim']),
            nn.Sigmoid()
        )
        
        self.attention_W = nn.Linear(pool_kwargs['att_dim'], self.K)
        
        # Classifier head (matching DeepMILAttModel)
        self.classifier = nn.Linear(emb_dim, 1)

    def forward(self, features_batch, adj_mat=None, mask=None, return_att=False, return_loss=False, **kwargs):
        """
        Forward pass with proper return values for MuRCL and heatmap generation.
        """
        # Feature extraction
        X = self.feat_ext(features_batch)  # [B, N, emb_dim]

        # ABMIL attention computation
        A_V = self.attention_V(X)  # [B, N, att_dim]
        A_U = self.attention_U(X)  # [B, N, att_dim]
        A = self.attention_W(A_V * A_U)  # [B, N, K]

        # Apply masking if provided
        if mask is not None:
            mask_expanded = mask.unsqueeze(-1)  # [B, N, 1]
            A = A.masked_fill(~mask_expanded, -1e9)

        # Softmax over instances
        A = torch.nn.functional.softmax(A, dim=1)  # [B, N, K]

        # Weighted aggregation
        M = torch.bmm(A.transpose(1, 2), X)  # [B, K, emb_dim]
        
        # Final bag embedding (average across attention heads)
        bag_embedding = M.mean(dim=1)  # [B, emb_dim]
        
        if return_att:
            # Return attention weights for heatmap generation
            attention_weights = A.mean(dim=-1)  # [B, N] - averaged across heads
            return bag_embedding, attention_weights
        else:
            # For MuRCL: return bag_embedding and a different ppo_state
            # PPO state should be the pre-aggregation features (for policy learning)
            ppo_state = X.mean(dim=1)  # [B, emb_dim] - mean pooling for policy state
            return bag_embedding, ppo_state