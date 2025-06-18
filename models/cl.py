import torch
import torch.nn as nn
import logging
from typing import List, Tuple, Optional

# It's good practice to get the logger from the root to inherit settings
logger = logging.getLogger(__name__)

class CL(nn.Module):
    def __init__(self, encoder, projection_dim, n_features):
        super(CL, self).__init__()

        self.encoder = encoder  # This is the GraphAndMILPipeline
        self.n_features = n_features
        self.projection_dim = projection_dim

        # Renamed from projection_head to projector for consistency with other literature
        self.projector = nn.Sequential(
            nn.Linear(self.n_features, self.n_features, bias=False),
            nn.ReLU(),
            nn.Linear(self.n_features, projection_dim, bias=False)
        )

    def forward(self, views_data_list: List[Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor]], **kwargs):
        """
        Processes a list of views, where each view's data is already batched.

        Args:
            views_data_list (List[Tuple]): A list containing data for each view.
                                           Example for 2 views:
                                           [
                                               (feats_batch_view1, adjs_batch_view1, masks_batch_view1),
                                               (feats_batch_view2, adjs_batch_view2, masks_batch_view2)
                                           ]
                                           - feats_batch: Tensor of shape [B, N, D_feat]
                                           - adjs_batch: Tensor of shape [B, N, N] or None
                                           - masks_batch: Tensor of shape [B, N]

        Returns:
            Tuple[List[torch.Tensor], List[torch.Tensor]]: 
                - A list of projected embeddings (z) for each view.
                - A list of bag embeddings (h) for each view.
        """
        z_projections = []
        h_bag_embeddings = []

        # This loop iterates over the two views
        for view_data_tuple in views_data_list:
            # Unpack the batched tensors for the current view
            features_batch, adj_mats_batch, masks_batch = view_data_tuple
            
            # self.encoder is the GraphAndMILPipeline, which expects batched tensors
            bag_embeddings_for_view, ppo_states_for_view = self.encoder(
                features_batch=features_batch, 
                adj_mats_batch=adj_mats_batch,
                masks_batch=masks_batch,
                **kwargs
            )
            
            # Project the bag embeddings for the contrastive loss
            projected_output = self.projector(bag_embeddings_for_view)
            
            z_projections.append(projected_output)
            # The bag embeddings are used as the state for the PPO agent
            h_bag_embeddings.append(ppo_states_for_view)
            
        return z_projections, h_bag_embeddings
