import torch
import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.nn as nn


class ChannelAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        B, C, H, W = x.shape
        x = x.permute(0, 2, 3, 1).reshape(-1, H * W, C)  # 调整形状以适应 qkv 的计算

        # 得到 query, key 和 value，在通道维度上进行线性投射
        qkv = self.qkv(x).reshape(B, -1, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        k = k * self.scale
        attention = (k @ v.transpose(-1, -2)).softmax(dim=-1)
        x = (attention @ q).transpose(1, 2).reshape(B, H, W, C)
        x = x.permute(0, 3, 1, 2)
        x = self.proj(x.reshape(B, -1, C)).reshape(B, C, H, W)  # 恢复到原始形状

        return x




class ChannelAttentionSequence(nn.Module):
    def __init__(self, emb_dim, num_heads=8, qkv_bias=True, att_dropout=0.0, dropout=0.0):
        super(ChannelAttentionSequence, self).__init__()
        self.emb_dim = emb_dim
        self.num_heads = num_heads
        self.head_dim = emb_dim // num_heads

        self.Wq = nn.Linear(emb_dim, emb_dim, bias=qkv_bias)
        self.Wk = nn.Linear(emb_dim, emb_dim, bias=qkv_bias)
        self.Wv = nn.Linear(emb_dim, emb_dim, bias=qkv_bias)

        self.proj_out = nn.Linear(emb_dim, emb_dim)

        self.dropout = nn.Dropout(dropout)
        self.att_dropout = nn.Dropout(att_dropout)

    def forward(self, x, context, pad_mask=None):
        '''
        :param x: [batch_size, seq_len, emb_dim]
        :param context: [batch_size, seq_len, emb_dim]
        :param pad_mask: [batch_size, seq_len, seq_len]
        :return: out, att_weights
        '''
        B, N, C = x.size()
        Q = self.Wq(x)  # [B, N, C]
        K = self.Wk(context)  # [B, N, C]
        V = self.Wv(context)  # [B, N, C]

        # Reshape Q, K, V for multi-head attention
        Q = Q.view(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)  # [B, num_heads, N, head_dim]
        K = K.view(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)  # [B, num_heads, N, head_dim]
        V = V.view(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)  # [B, num_heads, N, head_dim]

        # Calculate attention scores
        att_weights = torch.matmul(Q.transpose(2, 3), K) / (self.head_dim ** 0.5)  # [B, num_heads, head_dim, head_dim]

        # Apply mask if provided
        if pad_mask is not None:
            att_weights = att_weights.masked_fill(pad_mask.unsqueeze(1).unsqueeze(2), -1e9)

        att_weights = F.softmax(att_weights, dim=-1)  # [B, num_heads, head_dim, head_dim]
        att_weights = self.att_dropout(att_weights)

        # Weighted sum of values
        out = torch.matmul(att_weights, V.transpose(2, 3))  # [B, num_heads, head_dim, N]

        # Combine heads and project output
        out = out.permute(0, 3, 1, 2).contiguous()  # [B, N, num_heads, head_dim]
        out = out.view(B, N, C)  # [B, N, emb_dim]
        out = self.dropout(out)
        out = self.proj_out(out)

        return out, att_weights

import time
start = time.time()
# 示例用法
B, C, H, W = 3, 256, 128, 128
x = torch.randn(B, C, H, W)


channel_attention = ChannelAttention(dim=C, num_heads=8)
output = channel_attention(x)
print(time.time() - start)
print("Output shape: ", output.shape)

# # Example usage
# x = torch.randn(3, 256, 1024)  # [batch_size, seq_len, emb_dim]
# context = torch.randn(3, 256, 1024)  # [batch_size, seq_len, emb_dim]
#
# model = ChannelAttentionSequence(emb_dim=1024, num_heads=8)
# out, att_weights = model(x, context)
# print(out.shape)  # Expected output: [3, 10, 64]
# print(att_weights.shape)  # Expected output: [3, 8, 8, 8]
