from .common_component import *
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from einops import rearrange


#Codes are implemented based on the next papers to develop a new deep joint source channel coding system.
#[1] FAJSCC
#[2] X. Kong, H. Zhao, Y. Qiao, and C. Dong, ¡°ClassSR: A general framework to accelerate super-resolution networks by data characteristic,¡± 
#in Proc. IEEE Conf. Comput. Vis. Pattern Recognit., Jun. 2021, pp. 12016-12025.
#[3] Y. Wang, Y. Liu, S. Zhao, J. Li, and L. Zhang, ¡°CAMixerSR: Only details need more ¡°attention¡±,¡± 
#in Proc. IEEE Conf. Comput. Vis. Pattern Recognit., Jun. 2024, pp. 25837-25846.
#[4] S. W. Zamir, A. Arora, S. Khan, M. Hayat, F. S. Khan, and M.-H. Yang, ¡°Restormer: Efficient transformer for high-resolution image restoration,¡±
#in Proc. IEEE Conf. Comput. Vis. Pattern Recognit., Jun. 2022, pp. 5728-5739.
#[5] K. Yang, S. Wang, J. Dai, X. Qin, K. Niu, and P. Zhang, ¡°SwinJSCC: Taming swin transformer for deep joint source-channel coding,¡± 
#IEEE Trans. on Cogn. Commun. Netw., vol. 11, no. 1, pp. 90-104, Jul. 2024.


class DSC(nn.Module):
    def __init__(self, channels, kernel_size=(3, 3), stride=1):
        super().__init__()

        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)

        kH, kW = kernel_size

        # SAME padding to preserve H and W after conv
        pad_h = ((stride - 1) + (kH - 1)) // 2
        pad_w = ((stride - 1) + (kW - 1)) // 2

        #depth-wise convolution
        self.dconv = nn.Conv2d(in_channels=channels,out_channels=channels,kernel_size=(kH, kW),stride=stride,
            padding=(pad_h, pad_w), groups=channels)

        #point-wise convolution
        self.pconv = nn.Conv2d(in_channels=channels,out_channels=channels,
            kernel_size=1,stride=1,padding=0)

    def forward(self, x):
        out = self.dconv(x)
        out = self.pconv(out)
        return out


class RDSC(nn.Module):
    def __init__(self, channels, kernel_size, stride=1):
        super().__init__()
        
        self.layer_sequence = nn.Sequential(
            DSC(channels, kernel_size, stride),
            nn.LeakyReLU(0.1, inplace=True),
            DSC(channels, kernel_size, stride),
            nn.LeakyReLU(0.1, inplace=True)
        )


    def forward(self, x):
        out = self.layer_sequence(x) + x        
        return out



class WindowTransformerBlock(nn.Module):

    def __init__(self, dim, input_resolution, num_heads, window_size=7, shift_size=0,qkv_bias=True, qk_scale=None):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        if min(self.input_resolution) <= self.window_size:
            # if window size is larger than input resolution, we don't partition windows
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.attn = WindowAttention(
            dim, window_size=to_2tuple(self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale)

        if self.shift_size > 0:
            # calculate attention mask for SW-MSA
            H, W = self.input_resolution
            img_mask = torch.zeros((1, H, W, 1))  # 1 H W 1
            h_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            w_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1

            mask_windows = window_partition(img_mask, self.window_size)  # nW, window_size, window_size, 1
            mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        else:
            attn_mask = None

        self.register_buffer("attn_mask", attn_mask)

    def forward(self, x):
        B,C,H,W = x.shape
        if self.input_resolution != (H,W):
            self.input_resolution = (H,W)
            self.update_mask()
        x = rearrange(x,'b c h w -> b (h w) c', h=H,w=W)    
        H, W = self.input_resolution
        B, L, C = x.shape

        x = x.view(B, H, W, C)

        # cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x

        # partition windows
        x_windows = window_partition(shifted_x, self.window_size)  # nW*B, window_size, window_size, C
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C
        B_, N, C = x_windows.shape

        # merge windows
        attn_windows = self.attn(x_windows,
                                 add_token=False,
                                 mask=self.attn_mask)
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)  # B H' W' C

        # reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x
        x = x.view(B, H * W, C)

        x = rearrange(x,'b (h w) c -> b c h w', h=H,w=W)

        return x

    def update_mask(self):
        if self.shift_size > 0:
            # calculate attention mask for SW-MSA
            H, W = self.input_resolution
            img_mask = torch.zeros((1, H, W, 1))  # 1 H W 1
            h_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            w_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1

            mask_windows = window_partition(img_mask, self.window_size)  # nW, window_size, window_size, 1
            mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
            self.attn_mask = attn_mask.cuda()
        else:
            pass


def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size
    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image
    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class WindowAttention(nn.Module):
    r""" Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.
    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, add_token=True, token_num=0, mask=None):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        B_, N, C = x.shape
        #important
        #print("x.shape:",x.shape)
        #print("self.qkv(x).shape:",self.qkv(x).shape)
        #print("self.num_heads:",self.num_heads)
        #Note that C / self.num_heads should be integer

        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))  # (N+1)x(N+1)

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww

        if add_token:
            attn[:, :, token_num:, token_num:] = attn[:, :, token_num:, token_num:] + relative_position_bias.unsqueeze(
                0)
        else:
            attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            if add_token:
                # padding mask matrix
                mask = F.pad(mask, (token_num, 0, token_num, 0), "constant", 0)
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x




class DWMixer(nn.Module): #Depthwise convolution + Window attention mixer
    def __init__(self, dim, input_resolution, num_heads, window_size=7, shift_size=0,qkv_bias=True, qk_scale=None):
        super().__init__()
        
        self.RDSC = RDSC(channels=dim//2,kernel_size=5)
        self.WindowTransformerBlock = WindowTransformerBlock(dim//2, input_resolution, num_heads, window_size=window_size, shift_size=0,qkv_bias=True, qk_scale=None)


    def forward(self, x):
        x1, x2 = torch.chunk(x, chunks=2, dim=1)

        out1 = self.RDSC(x1)
        out2 = self.WindowTransformerBlock(x2)

        out = torch.cat([out1, out2], dim=1)
        return out




class LICRFBlock(nn.Module):
    def __init__(self, dim, input_resolution, num_heads, window_size=7, shift_size=0,qkv_bias=True, qk_scale=None):
        super().__init__()
        
        
        self.DSC1 = DSC(channels=dim,kernel_size=5)
        self.DWMixer = DWMixer(dim, input_resolution, num_heads, window_size, shift_size,qkv_bias, qk_scale)
        self.DSC2 = DSC(channels=dim,kernel_size=5)


    def forward(self, x):
        out = self.DSC1(x)
        out = self.DWMixer(out)
        out = self.DSC2(out)
        out = out + x
        return out

class LICRFGroup(nn.Module):
    def __init__(self, n_feats, n_block, num_heads, window_size, 
                  input_resolution=(128,128),qkv_bias=True, qk_scale=None):
        super().__init__()
        self.input_resolution = input_resolution
        self.n_block = n_block
        self.blocks = nn.ModuleList([
                      LICRFBlock(dim=n_feats,
                                 input_resolution=(input_resolution[0], input_resolution[1]),
                                 num_heads=num_heads, window_size=window_size,
                                 shift_size=0 if (i % 2 == 0) else window_size // 2,
                                 qkv_bias=qkv_bias, qk_scale=qk_scale)
            for i in range(n_block)])


    def forward(self, x):
        for _, blk in enumerate(self.blocks):
            x = blk(x)
        return x



