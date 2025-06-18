import torch
import torch.nn as nn
from typing import Optional, Tuple
import logging

logger = logging.getLogger(__name__)

class GraphAndMILPipeline(nn.Module):
    """
    Coherent, fully-batched pipeline combining Graph processing with MIL aggregation.
    This version is vectorized to be efficient and compatible with DataParallel.
    """
    
    def __init__(self, 
                 input_dim: int,
                 graph_encoder: Optional[nn.Module] = None,
                 mil_aggregator: Optional[nn.Module] = None):
        super().__init__()
        self.input_dim = input_dim
        self.graph_encoder = graph_encoder
        self.mil_aggregator = mil_aggregator
        
        if graph_encoder is not None:
            self.post_graph_dim = getattr(graph_encoder, 'output_dim', input_dim)
        else:
            self.post_graph_dim = input_dim
        
        # Standardize how we get the output dimension from different MIL models
        self.output_dim = getattr(mil_aggregator, 'L', None) or \
                          getattr(mil_aggregator, 'output_dim', None) or \
                          self.post_graph_dim

    def forward(self, 
                features_batch: torch.Tensor, 
                adj_mats_batch: Optional[torch.Tensor] = None,
                masks_batch: Optional[torch.Tensor] = None,
                **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Process a batch of WSIs through the Graph + MIL pipeline using vectorized operations.

        Args:
            features_batch (torch.Tensor): Batched features of shape [B, N, D_in].
            adj_mats_batch (Optional[torch.Tensor]): Batched adjacency matrices of shape [B, N, N].
            masks_batch (Optional[torch.Tensor]): Batched boolean masks of shape [B, N].

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: 
                - Final bag embeddings, shape [B, D_out].
                - PPO states (intermediate bag embeddings), shape [B, D_out].
        """
        # Step 1: Graph Processing (if applicable)
        # The graph_encoder now receives the full batch tensor.
        if self.graph_encoder is not None:
            processed_features = self.graph_encoder(features_batch, adj_mats_batch)
        else:
            processed_features = features_batch

        # Step 2: MIL Aggregation (if applicable)
        # The mil_aggregator now receives the full batch tensor.
        if self.mil_aggregator is not None:
            # The aggregator must be vectorized and handle masks internally.
            # It should return (bag_embedding, ppo_state).
            bag_embeddings, ppo_states = self.mil_aggregator(
                processed_features, 
                adj_mat=adj_mats_batch, 
                mask=masks_batch,
                return_attn=True # Ensure we get the PPO state back
            )
        else:
            # Fallback: If no MIL aggregator, use masked mean pooling.
            if masks_batch is not None:
                mask_expanded = masks_batch.unsqueeze(-1)
                masked_features = processed_features * mask_expanded
                summed_features = torch.sum(masked_features, dim=1)
                num_valid = masks_batch.sum(dim=1, keepdim=True).clamp(min=1)
                bag_embeddings = summed_features / num_valid
            else:
                bag_embeddings = torch.mean(processed_features, dim=1)
            
            ppo_states = bag_embeddings

        return bag_embeddings, ppo_states

def create_pipeline(input_dim: int, 
                   graph_encoder_type: str = "none",
                   mil_aggregator_type: str = "abmil",
                   **kwargs) -> GraphAndMILPipeline:
    """
    Factory function to create GraphAndMILPipeline with specified components.
    """
    graph_encoder = None
    if graph_encoder_type.lower() == "gat":
        try:
            from .graph_encoders import create_gat_encoder
            graph_encoder = create_gat_encoder(
                input_dim=input_dim,
                **kwargs
            )
            logger.info(f"Created GAT encoder with input_dim={input_dim}")
        except ImportError as e:
            logger.error(f"Failed to import or create GAT encoder: {e}")
        except Exception as e:
            logger.error(f"An error occurred creating GAT encoder: {e}", exc_info=True)

    mil_aggregator = None
    if mil_aggregator_type.lower() == "smtabmil":
        try:
            from .smtabmil import create_smtabmil
            # The MIL aggregator's input dim must match the graph encoder's output dim
            post_graph_dim = getattr(graph_encoder, 'output_dim', input_dim)
            mil_aggregator = create_smtabmil(
                feature_dim=post_graph_dim,
                **kwargs
            )
            logger.info(f"Created SmTABMIL aggregator with feature_dim={post_graph_dim}")
        except ImportError as e:
            logger.error(f"Failed to import or create SmTABMIL aggregator: {e}")
        except Exception as e:
            logger.error(f"An error occurred creating SmTABMIL aggregator: {e}", exc_info=True)
    
    elif mil_aggregator_type.lower() == "abmil":
        try:
            from .abmil import ABMIL
            post_graph_dim = getattr(graph_encoder, 'output_dim', input_dim)
            mil_aggregator = ABMIL(
                dim_in=post_graph_dim,
                **kwargs
            )
            logger.info(f"Created ABMIL aggregator with dim_in={post_graph_dim}")
        except ImportError as e:
            logger.error(f"Failed to import or create ABMIL aggregator: {e}")
        except Exception as e:
            logger.error(f"An error occurred creating ABMIL aggregator: {e}", exc_info=True)

    return GraphAndMILPipeline(
        input_dim=input_dim,
        graph_encoder=graph_encoder,
        mil_aggregator=mil_aggregator
    )