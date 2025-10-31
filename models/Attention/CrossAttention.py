import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

# bottle two
class CrossAttention(nn.Module):
    def __init__(self, in_channels, emb_dim, num_heads=8, qkv_bias=True, att_dropout=0.0, dropout=0.0):
        super(CrossAttention, self).__init__()
        self.emb_dim = emb_dim
        self.num_heads = num_heads
        self.head_dim = emb_dim // num_heads

        self.scale = self.head_dim ** -0.5

        self.proj_in1 = nn.Conv2d(in_channels, emb_dim, kernel_size=1, stride=1, padding=0)
        self.proj_in2 = nn.Conv2d(in_channels, emb_dim, kernel_size=1, stride=1, padding=0)
        # self.proj_in1 = nn.Conv2d(in_channels, emb_dim, kernel_size=4, stride=4, padding=0)
        # self.proj_in2 = nn.Conv2d(in_channels, emb_dim, kernel_size=4, stride=4, padding=0) # 结果会很平滑，对于细小血管带来负面影响

        self.Wq = nn.Linear(emb_dim, emb_dim, bias=qkv_bias)
        self.Wk = nn.Linear(emb_dim, emb_dim, bias=qkv_bias)
        self.Wv = nn.Linear(emb_dim, emb_dim, bias=qkv_bias)

        self.proj_out = nn.Conv2d(emb_dim, in_channels, kernel_size=1, stride=1, padding=0)
        self.deconv_layer = nn.ConvTranspose2d(emb_dim, in_channels, kernel_size=4, stride=4, padding=0)
        self.dropout = nn.Dropout(dropout)
        self.att_dropout = nn.Dropout(att_dropout)

    def forward(self, imgOne, imgTwo, pad_mask=None):
        '''
        :param imgOne: [batch_size, c, h, w]
        :param imgTwo: [batch_size, c, h/2, w/2]
        :param pad_mask: [batch_size, seq_len, seq_len]
        :return: out, att_weights
        '''
        b, c, h, w = imgOne.shape
        _, _, h2, w2 = imgTwo.shape

        # 将输入图像特征投影到嵌入维度
        x = self.proj_in1(imgOne)  # [batch_size, emb_dim, h, w]
        context = self.proj_in2(imgTwo)  # [batch_size, emb_dim, h2, w2]

        # b, c, h, w = x.shape
        # _, _, h2, w2 = context.shape
        # 将x和context展开
        x = rearrange(x, 'b c h w -> b (h w) c')  # [batch_size, h*w, emb_dim]
        context = rearrange(context, 'b c h w -> b (h w) c')  # [batch_size, (h/2)*(w/2), emb_dim]

        # 生成查询（Q）、键（K）和值（V）
        Q = self.Wq(x)  # [batch_size, h*w, emb_dim]
        K = self.Wk(context)  # [batch_size, (h/2)*(w/2), emb_dim]
        V = self.Wv(context)  # [batch_size, (h/2)*(w/2), emb_dim]

        # Splitting heads
        Q = Q.view(b, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)  # [B, num_heads, N1, head_dim]
        K = K.view(b, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)  # [B, num_heads, N2, head_dim]
        V = V.view(b, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)  # [B, num_heads, N2, head_dim]

        # 计算注意力权重
        att_weights = torch.matmul(Q, K.transpose(-2, -1)) * self.scale  # [B, num_heads, N1, N2]

        # 如果提供了 pad_mask，则屏蔽特定位置的权重
        if pad_mask is not None:
            att_weights = att_weights.masked_fill(pad_mask.unsqueeze(1), -1e9)

        # 计算 softmax 注意力权重
        att_weights = F.softmax(att_weights, dim=-1)  # [B, num_heads, N1, N2]
        att_weights = self.att_dropout(att_weights)

        # 使用注意力权重对 V 进行加权求和
        out = torch.matmul(att_weights, V)  # [B, num_heads, N1, head_dim]

        # Combine heads and project output
        out = out.permute(0, 2, 1, 3).contiguous()  # [B, N1, num_heads, head_dim]
        out = out.view(b, h * w, self.emb_dim)  # [B, N1, emb_dim]
        out = self.dropout(out)

        # 将输出特征重新排列回 [batch_size, c, h, w] 形状
        out = rearrange(out, 'b (h w) c -> b c h w', h=h, w=w)  # [batch_size, emb_dim, h, w]

        # 将输出特征重新投影回输入的通道数
        out = self.proj_out(out)  # [batch_size, c, h, w]
        # out = self.deconv_layer(out)  # [batch_size, c, h, w]

        out = out + imgOne

        # return out, att_weights
        return out

import torch
import torch.nn as nn
import torch.nn.functional as F

class CrossAttentionSequence(nn.Module):
    def __init__(self, emb_dim, num_heads=8, qkv_bias = True, att_dropout=0.0, dropout=0.0):
        super(CrossAttentionSequence, self).__init__()
        self.emb_dim = emb_dim
        self.num_heads = num_heads
        self.head_dim = emb_dim // num_heads

        self.Wq = nn.Linear(emb_dim, emb_dim, qkv_bias)
        self.Wk = nn.Linear(emb_dim, emb_dim, qkv_bias)
        self.Wv = nn.Linear(emb_dim, emb_dim, qkv_bias)

        self.proj_out = nn.Linear(emb_dim, emb_dim)

        self.dropout = nn.Dropout(dropout)
        self.att_dropout = nn.Dropout(att_dropout)

    def forward(self, x, context, pad_mask=None):
        '''
        :param x: [batch_size, seq_len1, emb_dim]
        :param context: [batch_size, seq_len2, emb_dim]
        :param pad_mask: [batch_size, seq_len1, seq_len2]
        :return: out, att_weights
        '''
        B, N1, C = x.size()
        _, N2, _ = context.size()

        Q = self.Wq(x)  # [B, N1, C]
        K = self.Wk(context)  # [B, N2, C]
        V = self.Wv(context)  # [B, N2, C]

        # Splitting heads
        Q = Q.view(B, N1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)  # [B, num_heads, N1, head_dim]
        K = K.view(B, N2, self.num_heads, self.head_dim).permute(0, 2, 1, 3)  # [B, num_heads, N2, head_dim]
        V = V.view(B, N2, self.num_heads, self.head_dim).permute(0, 2, 1, 3)  # [B, num_heads, N2, head_dim]

        # Attention scores
        att_weights = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)  # [B, num_heads, N1, N2]

        # Apply mask if provided
        if pad_mask is not None:
            att_weights = att_weights.masked_fill(pad_mask.unsqueeze(1), -1e9)

        att_weights = F.softmax(att_weights, dim=-1)  # [B, num_heads, N1, N2]
        att_weights = self.att_dropout(att_weights)

        # Weighted sum of values
        out = torch.matmul(att_weights, V)  # [B, num_heads, N1, head_dim]

        # Combine heads and project output
        out = out.permute(0, 2, 1, 3).contiguous()  # [B, N1, num_heads, head_dim]
        out = out.view(B, N1, C)  # [B, N1, emb_dim]
        out = self.dropout(out)
        out = self.proj_out(out)

        # return out, att_weights
        return out

# 示例用法
# if __name__ == "__main__":
#     model = CrossAttentionSequence(emb_dim=2048, num_heads=8)
#     x = torch.randn(3, 256, 2048)  # [batch_size, seq_len1, emb_dim]
#     context = torch.randn(3, 64, 2048)  # [batch_size, seq_len2, emb_dim]
#     output, att_weights = model(x, context)
#     print("Output shape:", output.shape)
#     print("Attention weights shape:", att_weights.shape)



# # 示例使用
if __name__ == '__main__':
    cross_attention = CrossAttention(in_channels=256, emb_dim=256)
    cross_attention1 = CrossAttention(in_channels=256, emb_dim=256)

    # 创建一个形状类似于三角形的矩阵
    def create_triangular_tensor(rows):
        triangle = []
        for i in range(1, rows + 1):
            row = np.zeros(rows)
            row[:i] = 1  # 填充前i个元素为1
            triangle.append(row)
        return np.array(triangle)
    def create_triangular_tensor_c(rows):
        triangle = []
        for i in range(1, rows + 1):
            row = np.zeros(rows)
            row[i:] = 1  # 填充前i个元素为1
            triangle.append(row)
        return np.array(triangle)

    # 设置行数
    rows = 128
    triangular_matrix = create_triangular_tensor(rows)
    triangular_matrix_c = create_triangular_tensor_c(64)

    # 将 NumPy 数组转换为 PyTorch 张量
    triangular_tensor = torch.tensor(triangular_matrix, dtype=torch.float32)
    triangular_tensor_c = torch.tensor(triangular_matrix_c, dtype=torch.float32)
    x = triangular_tensor.unsqueeze(0).unsqueeze(0).expand(3, 256, rows, rows)
    context = triangular_tensor_c.unsqueeze(0).unsqueeze(0).expand(3, 256, 64, 64)
    # 生成随机输入数据
    # x = torch.randn(3, 256, 128, 128)  # [batch_size, c, h, w]
    # context = torch.randn(3, 256, 64, 64)  # [batch_size, c, h/2, w/2]

    out, weight = cross_attention(x, context)
    image = out[0, 4, :, :].detach().numpy()

    # 显示图像
    import matplotlib.pyplot as plt

    # 可视化
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))

    # 显示三角形张量
    im0 = ax[0].imshow(triangular_tensor, cmap='gray')
    ax[0].set_title('Triangular Tensor')
    fig.colorbar(im0, ax=ax[0])

    # 显示 batch 0, channel 5 的图像
    im1 = ax[1].imshow(image, cmap='gray')
    ax[1].set_title('Batch 0, Channel 5')
    fig.colorbar(im1, ax=ax[1])

    plt.show()
    # out, att_weights = cross_attention1(context, x)

