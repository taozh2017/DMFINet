import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

sys.path.append(os.path.dirname(__file__))
'''
sys.path.append(os.path.dirname(__file__))
动态修改 Python 模块搜索路径的方法，使 Python 能够找到同一目录下的模块。具体来说：
__file__ 是一个内置变量，包含当前模块的文件路径。
os.path.dirname(__file__) 获取当前模块所在目录的路径。
sys.path.append(path) 将指定的路径 path 添加到 sys.path，从而使 Python 在这个路径下搜索模块。
通过这种方式，可以确保在同一目录下的模块可以相互导入，即使没有将这个目录声明为一个包。
'''
from CrossAttention import CrossAttentionSequence
from drop import DropPath
from helpers import *
from weight_init import trunc_normal_


class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, img_size=64, patch_size=4, in_chans=128, embed_dim=2048):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        B, C, H, W = x.shape

        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x).flatten(2).transpose(1, 2)
        # 64 64 64  ->1024 16 16  -> 1024 256  -> 256 1024
        # 3 224 224 -> 768 14 14 -> 768 196 -> 196 768  总的数据量没变
        # 原始embed_dim的设置使得总的数据量没变，则我也先这么使用
        return x

class PatchUnembed(nn.Module):
    def __init__(self, img_size=64, patch_size=4, embed_dim=2048, in_chans=128):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.img_size = img_size
        self.patch_size = patch_size

        self.num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.embed_dim = embed_dim

        self.proj = nn.ConvTranspose2d(embed_dim, in_chans, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        B, N, C = x.shape
        # Assume x is of shape (batch_size, num_patches, embed_dim)
        # Reshape to (batch_size, embed_dim, height, width)
        x = x.transpose(1, 2).contiguous().view(B, C, self.img_size[0] // self.patch_size[0], self.img_size[1] // self.patch_size[1])
        # Apply transposed convolution to get back to the original image
        x = self.proj(x)
        return x

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = CrossAttentionSequence(emb_dim=dim, num_heads=8, qkv_bias = qkv_bias)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, y):
        x = x + self.drop_path(self.attn(self.norm1(x), y))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class CrossAttentionBlock(nn.Module):
    def __init__(self, img_size=64, img_size_y=32, patch_size=4, in_chans=128, num_classes=1000, embed_dim=2048, depth=3,
                 num_heads=8, mlp_ratio=4., qkv_bias=True, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., hybrid_backbone=None, norm_layer=nn.LayerNorm, patch_first=False, un_patch=False):
        super(CrossAttentionBlock, self).__init__()

        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches
        self.patch_first = patch_first
        self.un_patch = un_patch
        if self.un_patch:
            self.un_patch_embed = PatchUnembed(img_size=img_size, patch_size=patch_size, in_chans=in_chans,
                                               embed_dim=embed_dim)

        self.patch_embed_y = PatchEmbed(
            img_size=img_size_y, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches_y = self.patch_embed_y.num_patches

        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)
        self.pos_embed_y = nn.Parameter(torch.zeros(1, num_patches_y, embed_dim))
        self.pos_drop_y = nn.Dropout(p=drop_rate)

        self.blk = Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=drop_path_rate, norm_layer=norm_layer)

        trunc_normal_(self.pos_embed, std=.02)
        trunc_normal_(self.pos_embed_y, std=.02)


    def forward(self, x, y, pad_mask=None):
        if self.patch_first:
            x = self.patch_embed(x)
            x = x + self.pos_embed
            x = self.pos_drop(x)

        y = self.patch_embed_y(y)
        y = y + self.pos_embed_y
        y = self.pos_drop(y)

        x = self.blk(x, y)

        if self.un_patch:
            x = self.un_patch_embed(x)

        return x

if __name__ == "__main__":
    cat = CrossAttentionBlock(img_size=64, img_size_y=32, patch_size=4, in_chans=128, embed_dim=2048, depth=3, num_heads=8,
                    mlp_ratio=4., qkv_bias=True, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                     drop_path_rate=0., hybrid_backbone=None, norm_layer=nn.LayerNorm, patch_first=True)

    a = torch.randn((4, 128, 64, 64))
    b = torch.randn((4, 128, 32, 32))

    cat1 = CrossAttentionBlock(img_size=64, img_size_y=32, patch_size=4, in_chans=128, embed_dim=2048, depth=3, num_heads=8,
                    mlp_ratio=4., qkv_bias=True, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                     drop_path_rate=0., hybrid_backbone=None, norm_layer=nn.LayerNorm, patch_first=False, un_patch=True)
    c = cat(a, b)
    d = torch.randn((4, 128, 32, 32))
    e = cat1(c, d)




















