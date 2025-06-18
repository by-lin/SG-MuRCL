import torch
import torch.nn.functional as F
from torch import nn
from typing import Tuple, Optional

class ABMIL(nn.Module):
    def __init__(self, dim_in, L=512, D=128, K=1, dropout=0., **kwargs):
        super(ABMIL, self).__init__()
        self.dim_in = dim_in
        self.L = L
        self.D = D
        self.K = K
        self.output_dim = L

        self.encoder = nn.Sequential(
            nn.Linear(dim_in, L),
            nn.ReLU(),
        )

        self.attention = nn.Sequential(
            nn.Linear(L, D),
            nn.Tanh(),
            nn.Linear(D, K)
        )

    def forward(self, 
                features_batch: torch.Tensor, 
                mask: Optional[torch.Tensor] = None,
                **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Fully-batched forward pass for ABMIL.

        Args:
            features_batch (torch.Tensor): Batched features of shape [B, N, D_in].
            mask (Optional[torch.Tensor]): Batched boolean masks of shape [B, N].

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: 
                - Final bag embeddings, shape [B, L].
                - PPO states (same as bag embeddings), shape [B, L].
        """
        H = self.encoder(features_batch)  # [B, N, L]

        # Compute Attention Scores
        A_raw = self.attention(H)  # [B, N, K]

        # CRITICAL: Apply Masking
        # Mask out padded instances before softmax by setting their scores to a very low number.
        if mask is not None:
            # mask shape is [B, N], we need [B, N, 1] for broadcasting
            A_raw = A_raw.masked_fill(~mask.unsqueeze(-1), -1e9)

        # Compute Attention Weights
        A = F.softmax(A_raw, dim=1)  # Softmax over instances (N) -> [B, N, K]

        # Apply Attention
        # Transpose A for batch matrix multiplication: [B, K, N]
        # Multiply with H: [B, N, L] -> Result: [B, K, L]
        M = torch.bmm(A.transpose(1, 2), H)

        # Average across attention heads if K > 1
        bag_embedding = M.mean(dim=1) # [B, L]
        
        # For ABMIL, the PPO state is the final bag embedding
        ppo_state = bag_embedding

        return bag_embedding, ppo_state

def create_abmil(feature_dim, **kwargs):
    """Factory function to create ABMIL model."""
    return ABMIL(dim_in=feature_dim, **kwargs)