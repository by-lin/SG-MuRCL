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
                adj_mat: Optional[torch.Tensor] = None,
                mask: Optional[torch.Tensor] = None,
                **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Process features through Graph + MIL pipeline.
        
        Args:
            features_batch: [B, N, D] or for single sample [N, D]
            adj_mat: [B, N, N] or for single sample [N, N] 
            mask: [B, N] or for single sample [N]
        
        Returns:
            bag_embeddings: [B, emb_dim] or [emb_dim]
            ppo_states: [B, emb_dim] or [emb_dim]
        """
        # Handle single sample vs batch
        single_sample = features_batch.ndim == 2
        if single_sample:
            features_batch = features_batch.unsqueeze(0)
            if adj_mat is not None:
                adj_mat = adj_mat.unsqueeze(0)
            if mask is not None:
                mask = mask.unsqueeze(0)
        
        # Step 1: Graph Processing
        if self.graph_encoder is not None:
            processed_features = self.graph_encoder(features_batch, adj_mat)
        else:
            processed_features = features_batch

        # Step 2: MIL Aggregation
        if self.mil_aggregator is not None:
            bag_embeddings, ppo_states = self.mil_aggregator(
                processed_features, 
                adj_mat=adj_mat, 
                mask=mask
            )
        else:
            # Fallback
            if mask is not None:
                mask_expanded = mask.unsqueeze(-1)
                masked_features = processed_features * mask_expanded
                summed_features = torch.sum(masked_features, dim=1)
                num_valid = mask.sum(dim=1, keepdim=True).clamp(min=1)
                bag_embeddings = summed_features / num_valid
            else:
                bag_embeddings = torch.mean(processed_features, dim=1)
            
            ppo_states = bag_embeddings

        # Remove batch dimension if single sample
        if single_sample:
            bag_embeddings = bag_embeddings.squeeze(0)
            ppo_states = ppo_states.squeeze(0)

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