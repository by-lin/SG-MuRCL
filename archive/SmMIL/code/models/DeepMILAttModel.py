import torch

from .modules import TransformerEncoder, MILAttentionPool

class DeepMILAttModel(torch.nn.Module):
    def __init__(
        self,
        input_dim : int,
        emb_dim : int,
        transformer_encoder_kwargs : dict = {},
        pool_kwargs : dict = {},
        ce_criterion : torch.nn.Module = torch.nn.BCEWithLogitsLoss(),
        **kwargs        
        ) -> None:
        super(DeepMILAttModel, self).__init__()
        self.input_dim = input_dim
        self.emb_dim = emb_dim
        self.pool_kwargs = pool_kwargs
        self.transformer_encoder_kwargs = transformer_encoder_kwargs
        self.kwargs = kwargs
        self.ce_criterion = ce_criterion

        self.feat_ext = torch.nn.Sequential(
            torch.nn.Linear(self.input_dim, emb_dim),
            torch.nn.GELU()
        )

        if len(transformer_encoder_kwargs.keys()) > 0:
            self.transformer_encoder = TransformerEncoder(in_dim = self.emb_dim, **transformer_encoder_kwargs)
        else:
            self.transformer_encoder = None
        self.pool = MILAttentionPool(in_dim = self.emb_dim, **pool_kwargs)
        self.classifier = torch.nn.Linear(self.emb_dim, 1)

    
    def forward(self, X, adj_mat, mask, return_att=False, return_loss=False):
        """
        input:
            X: tensor (batch_size, bag_size, ...)
            adj_mat: sparse coo tensor (batch_size, bag_size, bag_size)
            mask: tensor (batch_size, bag_size)
        output:
            T_logits_pred: tensor (batch_size,)
            att: tensor (batch_size, bag_size) if return_att is True
        """

        X = self.feat_ext(X) # (batch_size, bag_size, D)
        if self.transformer_encoder is not None:
            X = self.transformer_encoder(X, adj_mat, mask)

        out_pool = self.pool(X, adj_mat, mask, return_att=return_att)
        if return_att:
            Z, f = out_pool # Z: (batch_size, D, n_samples), f: (batch_size, bag_size, n_samples)
            if len(f.shape) == 2:
                f = f.unsqueeze(dim=-1) 
            f = torch.mean(f, dim=2) # (batch_size, bag_size)
        else:
            Z = out_pool # (batch_size, D, n_samples)
        
        if len(Z.shape) == 2:
            Z = Z.unsqueeze(dim=-1) # (batch_size, D, 1)
        
        Z = Z.transpose(1,2) # (batch_size, n_samples, D)
        T_logits_pred = self.classifier(Z) # (batch_size, n_samples, 1)
        T_logits_pred = torch.mean(T_logits_pred, dim=(1,2)) # (batch_size,)

        if return_att:
            return T_logits_pred, f
        else:
            return T_logits_pred
    
    def compute_loss(self, T_labels, X, adj_mat, mask, *args, **kwargs):
        """
        Input:
            T_labels: tensor (batch_size,)
            X: tensor (batch_size, bag_size, ...)
            adj_mat: sparse coo tensor (batch_size, bag_size, bag_size)
            mask: tensor (batch_size, bag_size)
        Output:
            T_logits_pred: tensor (batch_size,)
            loss_dict: dict {'BCEWithLogitsLoss', ...}
        """

        T_logits_pred = self.forward(X, adj_mat, mask, return_att=False)
        ce_loss = self.ce_criterion(T_logits_pred.float(), T_labels.float())

        return T_logits_pred, { 'BCEWithLogitsLoss': ce_loss }
    
    @torch.no_grad()
    def predict(self, X, adj_mat, mask, *args, return_y_pred=True, **kwargs):
        T_logits_pred, att_val = self.forward(X, adj_mat, mask, return_att=return_y_pred)
        return T_logits_pred, att_val
        


        
        