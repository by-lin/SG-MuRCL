import torch
from .Sm import ExactSm, ApproxSm

class MILAttentionPool(torch.nn.Module):
    def __init__(
            self, 
            in_dim, 
            att_dim=50, 
            sm_alpha=None, 
            sm_mode='approx', 
            sm_steps=10,
            sm_where='early', 
            spectral_norm=False, 
            **kwargs
        ):
        super(MILAttentionPool, self).__init__()
        self.in_dim = in_dim
        self.att_dim = att_dim
        self.sm_alpha = sm_alpha
        self.sm_mode = sm_mode
        self.sm_steps = sm_steps
        self.sm_where = sm_where
        self.spectral_norm = spectral_norm

        self.fc1 = torch.nn.Linear(in_dim, att_dim)
        self.fc2 = torch.nn.Linear(att_dim, 1, bias=False)
        
        if self.sm_alpha not in [0, 0.0, None] and self.sm_where != 'none':
            if self.sm_mode == 'exact':
                print('Using ExactSm')
                self.sm_layer = ExactSm(alpha=self.alpha)
            else:
                print('Using ApproxSm with num_steps={}'.format(self.sm_steps))
                self.sm_layer = ApproxSm(alpha=self.sm_alpha, num_steps=self.sm_steps)
            
            if self.sm_where not in ['late', 'mid', 'early']:
                raise ValueError("sm_where must be 'none', 'late', 'mid' or 'early'")

            if self.spectral_norm:
                if self.sm_where == 'mid':
                    self.fc2 = torch.nn.utils.parametrizations.spectral_norm(self.fc2)
                elif self.sm_where in ['early', 'none']:
                    self.fc1 = torch.nn.utils.parametrizations.spectral_norm(self.fc1)
                    self.fc2 = torch.nn.utils.parametrizations.spectral_norm(self.fc2)
            
            self.use_sm = True
        else:
            self.sm_layer = None
            self.use_sm = False
            self.sm_mode = ''
            self.sm_where = ''        
    
    def forward(self, X, adj_mat=None, mask=None, return_att=False):
        """
        input:
            X: tensor (batch_size, bag_size, D)
            adj_mat: sparse coo tensor (batch_size, bag_size, bag_size)
            mask: tensor (batch_size, bag_size)
        output:
            z: tensor (batch_size, D)
            s: tensor (bag_size, 1)
        """

        batch_size = X.shape[0]
        bag_size = X.shape[1]
        D = X.shape[2]
        
        if mask is None:
            mask = torch.ones(batch_size, bag_size, device=X.device)
        mask = mask.unsqueeze(dim=-1) # (batch_size, bag_size, 1)

        if self.sm_where == 'early' and self.use_sm:
            X = self.sm_layer(X, adj_mat) # (batch_size, bag_size, D)

        H = self.fc1(X.reshape(-1, D)).view(batch_size, bag_size, -1) # (batch_size, bag_size, att_dim)
        if self.sm_where == 'mid' and self.use_sm:
            H = self.sm_layer(H, adj_mat) # (batch_size, bag_size, att_dim)
        H = torch.nn.functional.tanh(H) # (batch_size, bag_size, L)

        f = self.fc2(H.reshape(-1, self.att_dim)).view(batch_size, bag_size, -1) # (batch_size, bag_size, 1)
        # f = f / math.sqrt(self.in_dim) # (batch_size, bag_size, 1)
        if self.sm_where =='late' and self.use_sm:
            f = self.sm_layer(f, adj_mat) # (batch_size, bag_size, 1)

        exp_f = torch.exp(f)*mask # (batch_size, bag_size, 1)
        sum_exp_f = torch.sum(exp_f, dim=1, keepdim=True) # (batch_size, 1, 1)
        s = exp_f/sum_exp_f # (batch_size, bag_size, 1)
        z = torch.bmm(X.transpose(1,2), s).squeeze(dim=2) # (batch_size, D)

        if return_att:
            return z, f.squeeze(dim=2)
        else:
            return z
        
    def compute_loss(self, *args, **kwargs):
        return {}