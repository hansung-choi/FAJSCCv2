from .common_component import *
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from einops import rearrange


#Components for ablation study of our proposed FAJSCC

class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first. 
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with 
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs 
    with shape (batch_size, channels, height, width).
    """
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_first"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError 
        self.normalized_shape = (normalized_shape, )
    
    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x

class ElementScale(nn.Module):
    """A learnable element-wise scaler."""

    def __init__(self, embed_dims, init_value=0., requires_grad=True):
        super(ElementScale, self).__init__()
        self.scale = nn.Parameter(
            init_value * torch.ones((1, embed_dims, 1, 1)),
            requires_grad=requires_grad
        )

    def forward(self, x):
        return x * self.scale

def ones(tensor):
    if tensor is not None:
        tensor.data.fill_(0.5)

def zeros(tensor):
    if tensor is not None:
        tensor.data.fill_(0.0)


#need to implement get_spatial_mask method.
class LAPredictor(nn.Module):
    """ Tree structured attention family predictor """
    def __init__(self, dim,k=4):
        super().__init__()
        cdim = dim + k
        mid_dim = cdim // 4

        # Root backbone
        self.backbone = nn.Sequential(
            nn.Conv2d(cdim, mid_dim, 1),
            LayerNorm(mid_dim),  # Replace with nn.GroupNorm if necessary
            nn.LeakyReLU(0.1, inplace=True)
        )

        # Channel Attention path
        self.ca_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(mid_dim, dim, 1),
            nn.Sigmoid()
        )

        # Shared branch for offset, SA, and mask
        self.shared_branch = nn.Sequential(
            nn.Conv2d(mid_dim, mid_dim, 3, padding=1),
            nn.LeakyReLU(0.1, inplace=True)
        )

        # Intermediate attention branch
        self.attn_branch = nn.Sequential(
            nn.Conv2d(mid_dim, mid_dim, 3, padding=1),
            nn.LeakyReLU(0.1, inplace=True)
        )

        # Spatial Attention head
        self.sa_head = nn.Sequential(
            nn.Conv2d(mid_dim, 1, 3, padding=1),
            nn.Sigmoid()
        )


    def forward(self, input_x):
        x_root = self.backbone(input_x)

        # Channel Attention
        ca = self.ca_head(x_root)

        # Shared computation
        x_shared = self.shared_branch(x_root)

        # Spatial attention path
        x_attn = self.attn_branch(x_shared)
        sa = self.sa_head(x_attn)

        return ca, sa


#need to implement get_spatial_mask method.
class LAmixer(nn.Module):
    def __init__(self, dim, bias=True, is_deformable=True,window_size=8):
        super().__init__()    

        self.dim = dim
        self.is_deformable = is_deformable
        self.window_size = window_size

        k = 3
        d = 2

        self.project_v = nn.Conv2d(dim, dim, 1, 1, 0, bias = bias)

        # Conv
        self.conv_sptial = nn.Sequential(
            nn.Conv2d(dim, dim, k, padding=k//2, groups=dim),
            nn.Conv2d(dim, dim, k, stride=1, padding=((k//2)*d), groups=dim, dilation=d))        
        self.project_out = nn.Conv2d(dim, dim, 1, 1, 0, bias = bias)

        self.act = nn.GELU()
        # Predictor
        self.route = LAPredictor(dim)

    def forward(self,x,condition_global=None, mask=None, train_mode=False):
        N,C,H,W = x.shape

        v = self.project_v(x)

        if self.is_deformable:
            condition_wind = torch.stack(torch.meshgrid(torch.linspace(-1,1,self.window_size),torch.linspace(-1,1,self.window_size)))\
                    .type_as(x).unsqueeze(0).repeat(N, 1, H//self.window_size, W//self.window_size)
            if condition_global is None:
                _condition = torch.cat([v, condition_wind], dim=1)
            else:
                _condition = torch.cat([v, condition_global, condition_wind], dim=1)

        ca, sa = self.route(_condition) #mask.size() = B X N X 1, N=(h w)
        v = v*sa
        
        out = v
        out = self.act(self.conv_sptial(out))*ca + out
        out = self.project_out(out)

        return out




class GatedFeedForward(nn.Module):
    def __init__(self, dim, mult = 1, bias=False, dropout = 0.):
        super().__init__()
        self.dim = dim

        self.project_in = nn.Conv2d(dim, dim*2, kernel_size=1, bias=bias)
        self.dwconv = nn.Conv2d(dim*2, dim*2, kernel_size=3, stride=1, padding=1, groups=dim*2, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x


class LABlock(nn.Module):
    def __init__(self, n_feats,window_size):
        super(LABlock,self).__init__()
        
        self.n_feats = n_feats
        self.norm1 = LayerNorm(n_feats)
        self.mixer = LAmixer(n_feats,window_size)
        self.norm2 = LayerNorm(n_feats)
        self.ffn = GatedFeedForward(n_feats)
        
    def forward(self,x,condition_global=None):
        res = self.mixer(x,condition_global)
        x = self.norm1(x+res)
        res = self.ffn(x)
        x = self.norm2(x+res)
        return x 

 
    

class LAGroup(nn.Module):
    def __init__(self, n_feats, n_block,window_size):
        super(LAGroup, self).__init__()
        
        self.n_feats = n_feats
        self.n_block = n_block
        self.global_predictor = nn.Sequential(nn.Conv2d(n_feats, 8, 1, 1, 0, bias=True),
                                        nn.LeakyReLU(negative_slope=0.1, inplace=True),
                                        nn.Conv2d(8, 2, 3, 1, 1, bias=True),
                                        nn.LeakyReLU(negative_slope=0.1, inplace=True))

        self.body = nn.ModuleList([LABlock(n_feats,window_size) for i in range(n_block)])
        self.body_tail = nn.Conv2d(n_feats, n_feats, 1, 1, 0)
        
    def forward(self,x):
        decision = []
        condition_global = self.global_predictor(x)
        shortcut = x.clone()
        for _, blk in enumerate(self.body):
            x = blk(x,condition_global)
        x = self.body_tail(x) + shortcut
        return x 





