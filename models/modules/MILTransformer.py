import torch

from torch.nn.attention import SDPBackend

from .Sm import ExactSm, ApproxSm

SDP_BACKEND = [SDPBackend.MATH, SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION, SDPBackend.CUDNN_ATTENTION]
# SDP_BACKEND = [SDPBackend.EFFICIENT_ATTENTION]

# Implementation follows SimpleViT from https://github.com/lucidrains/vit-pytorch/blob/main/vit_pytorch/simple_vit.py

class MultiheadSelfAttention(torch.nn.Module):
    def __init__(
            self,     
            att_dim, 
            in_dim=None, 
            num_heads=4, 
            dropout=0.0, 
            use_sm=False, 
            sm_alpha=0.0, 
            sm_mode='approx',
            sm_steps=10,
        ):
        super(MultiheadSelfAttention, self).__init__()
        self.att_dim = att_dim
        self.in_dim = in_dim
        if self.in_dim is None:
            self.in_dim = att_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.use_sm = use_sm
        self.sm_alpha = sm_alpha
        self.sm_mode = sm_mode
        self.sm_steps = sm_steps

        self.head_dim = self.att_dim // num_heads
        self.qkv_nn = torch.nn.Linear(self.in_dim, 3 * self.att_dim, bias = False)

        if self.use_sm:
            if self.sm_mode == 'approx':
                self.sm_layer = ApproxSm(alpha=self.sm_alpha, num_steps=self.sm_steps)
            elif self.sm_mode == 'exact':
                self.sm_layer = ExactSm(alpha=self.sm_alpha)
            else:
                raise ValueError(f"[MultiheadSelfAttention] Unknown smooth mode: {self.sm_mode}")

    def _scaled_dot_product_attention(self, query, key, value, mask=None):
        """
        input:
            query: (batch_size, num_heads, seq_len, head_dim)
            key: (batch_size, num_heads, seq_len, head_dim)
            value: (batch_size, num_heads, seq_len, head_dim)
            mask: (batch_size, seq_len)
        output:
            out: (batch_size, num_heads, seq_len, head_dim)       
        """
        mask = mask.unsqueeze(1).unsqueeze(1) # (batch_size, 1, 1, seq_len)
        mask = mask.repeat(1, 1, query.size(2), 1).bool() # (batch_size, num_heads, seq_len, seq_len)
         
        with torch.nn.attention.sdpa_kernel(SDP_BACKEND):
            out = torch.nn.functional.scaled_dot_product_attention(query, key, value, mask, self.dropout)
        
        return out

    def _qkv(self, x):
        """
        input:
            x: (batch_size, seq_len, in_dim)
        output:
            query: (batch_size, seq_len, att_dim)
            key: (batch_size, seq_len, att_dim)
            value: (batch_size, seq_len, att_dim)        
        """
        q, k, v = self.qkv_nn(x).chunk(3, dim=-1) # (batch_size, seq_len, att_dim), (batch_size, seq_len, att_dim), (batch_size, seq_len, att_dim)
        return q, k, v

    def forward(self, x, adj_mat=None, mask=None):
        """
        input:
            x: (batch_size, seq_len, in_dim)
            adj_mat: sparse coo tensor (batch_size, seq_len, seq_len)
            mask: (batch_size, seq_len)
        output:
            y: (batch_size, seq_len, att_dim)
        """
        batch_size, seq_len, in_dim = x.size()
        query, key, value = self._qkv(x) # (batch_size, seq_len, att_dim), (batch_size, seq_len, att_dim), (batch_size, seq_len, att_dim)
        query = query.view(batch_size, seq_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3) # (batch_size, num_heads, seq_len, head_dim)
        key = key.view(batch_size, seq_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3) # (batch_size, num_heads, seq_len, head_dim)
        value = value.view(batch_size, seq_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3) # (batch_size, num_heads, seq_len, head_dim)
        y = self._scaled_dot_product_attention(query, key, value, mask) # (batch_size, num_heads, seq_len, att_dim)
        y = y.permute(0, 2, 1, 3).contiguous().view(batch_size, seq_len, self.att_dim) # (batch_size, seq_len, att_dim)
        
        if self.use_sm:
            y = self.sm_layer(y, adj_mat)

        return y
    
class Identity(torch.nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x

class TransformerEncoderLayer(torch.nn.Module):
    def __init__(
            self, 
            att_dim,
            in_dim=None,
            num_heads=4,use_ff=True,
            dropout=0.0,
            use_sm=False, 
            sm_alpha=0.0, 
            sm_mode='approx',
            sm_steps=10,
        ):
        super(TransformerEncoderLayer, self).__init__()
        self.att_dim = att_dim
        self.in_dim = in_dim
        if self.in_dim is None:
            self.in_dim = att_dim
        self.num_heads = num_heads
        self.use_ff = use_ff
        self.dropout = dropout
        self.use_sm = use_sm
        self.sm_alpha = sm_alpha
        self.sm_mode = sm_mode
        self.sm_steps = sm_steps
        
        self.mha_layer = MultiheadSelfAttention(att_dim, in_dim, num_heads, dropout=dropout, use_sm=use_sm, sm_alpha=sm_alpha, sm_mode=sm_mode, sm_steps=sm_steps)

        if self.use_ff:
            self.ff_layer = torch.nn.Sequential(
                torch.nn.Linear(att_dim, att_dim),
                torch.nn.GELU(),
                torch.nn.Linear(att_dim, att_dim)
                )
        else:
            self.ff_layer = Identity()
        
        self.norm1 = torch.nn.LayerNorm(att_dim)
        self.norm2 = torch.nn.LayerNorm(att_dim)


    def forward(self, X, adj_mat=None, mask=None, **kwargs):
        """
        input:
            X: (batch_size, bag_size, in_dim)
            adj_mat: (batch_size, bag_size, bag_size)
            mask: (batch_size, bag_size)
        output:
            Y: (batch_size, bag_size, in_dim)
            
        """
        Y = self.mha_layer(self.norm1(X), adj_mat=adj_mat, mask=mask) + X # (batch_size, bag_size, att_dim)
        if self.use_ff:
            Y = self.ff_layer(self.norm2(Y)) + Y # (batch_size, bag_size, att_dim)
        return Y

class TransformerEncoder(torch.nn.Module):
    def __init__(
            self, 
            in_dim, 
            att_dim, 
            num_heads, 
            num_layers, 
            use_ff=False, 
            dropout=0.0, 
            use_sm=False, 
            sm_alpha=0.0, 
            sm_mode='approx',
            sm_steps=10,
        ):
        super(TransformerEncoder, self).__init__()
        self.in_dim = in_dim
        self.att_dim = att_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.use_ff = use_ff
        self.dropout = dropout
        self.use_sm = use_sm
        self.sm_alpha = sm_alpha
        self.sm_mode = sm_mode
        self.sm_steps = sm_steps

        if in_dim != att_dim:
            self.in_proj = torch.nn.Linear(in_dim, att_dim)
            self.out_proj = torch.nn.Linear(att_dim, in_dim)
        else:
            self.in_proj = Identity()
            self.out_proj = Identity()
        self.transf_layers = torch.nn.ModuleList([TransformerEncoderLayer(att_dim, att_dim, num_heads, use_ff=use_ff, dropout=dropout, use_sm=use_sm, sm_alpha=sm_alpha, sm_mode=sm_mode, sm_steps=sm_steps) for _ in range(num_layers)])
        
        self.norm = torch.nn.LayerNorm(att_dim)

    def forward(self, X, adj_mat=None, mask=None, **kwargs):
        """
        input:
            X: (batch_size, bag_size, in_dim)
            adj_mat: (batch_size, bag_size, bag_size)
            mask: (batch_size, bag_size)
        output:
            Y: (batch_size, bag_size, in_dim)
        """

        X = self.in_proj(X) # (batch_size, bag_size, att_dim)
        for layer in self.transf_layers:
            X = layer(X, adj_mat=adj_mat, mask=mask, **kwargs) # (batch_size, bag_size, att_dim)
        X = self.norm(X) # (batch_size, bag_size, att_dim)
        X = self.out_proj(X)
        
        return X, None