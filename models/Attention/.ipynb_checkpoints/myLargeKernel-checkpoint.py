import torch
import torch.nn as nn
from functools import reduce
from models.Attention.NolLocal import NonLocalBlock
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

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

    def forward(self, local, gl):
        b, c, h, w = local.shape
        _, _, h2, w2 = gl.shape

        # 将输入图像特征投影到嵌入维度
        x = self.proj_in1(local)  # [batch_size, emb_dim, h, w]
        context = self.proj_in2(gl)  # [batch_size, emb_dim, h2, w2]

        x = rearrange(x, 'b c h w -> b (h w) c')  # [batch_size, h*w, emb_dim]
        context = rearrange(context, 'b c h w -> b (h w) c')  # [batch_size, (h/2)*(w/2), emb_dim]

        # 生成查询（Q）、键（K）和值（V）
        Q = self.Wq(context)  # [batch_size, h*w, emb_dim]
        K = self.Wk(x)  # [batch_size, (h/2)*(w/2), emb_dim]
        V = self.Wv(x)  # [batch_size, (h/2)*(w/2), emb_dim]

        # Splitting heads
        Q = Q.view(b, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)  # [B, num_heads, N1, head_dim]
        K = K.view(b, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)  # [B, num_heads, N2, head_dim]
        V = V.view(b, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)  # [B, num_heads, N2, head_dim]

        # 计算注意力权重
        att_weights = torch.matmul(Q, K.transpose(-2, -1)) * self.scale  # [B, num_heads, N1, N2]

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

        out = out + local

        # return out, att_weights
        return out



class CAMEhance(nn.Module):
    def __init__(self, in_planes, ratio=8):
        super(CAMEhance, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes * 2, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()

    def forward(self, x, y):  # 256 64 64
        xy = torch.cat([x, y], dim=1)  # 512 64 64
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(xy))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(xy))))
        out = avg_out + max_out
        attn = self.sigmoid(out)
        out = x * attn + x

        return out

class LargeConv(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.conv0 = nn.Conv2d(dim, dim, 3, padding=1, groups=dim, dilation=1) # 3 9 19
        self.conv1 = nn.Conv2d(dim, dim, 3, padding=3, groups=dim, dilation=3)
        self.conv2 = nn.Conv2d(dim, dim, 3, padding=5, groups=dim, dilation=5)
        self.NL = NonLocalBlock(dim)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(dim * 6, dim // 2, 1, bias=False)
        self.fc1=nn.Sequential(nn.Conv2d(dim * 6, dim // 2,1,bias=False),
                               nn.ReLU(inplace=True))   # 降维
        self.fc2 = nn.Conv2d(dim // 2, dim * 3, kernel_size=1, bias=False)

        self.softmax = nn.Softmax(dim=1)  # 指定dim=1  使得两个全连接层对应位置进行softmax,保证 对应位置a+b+..=1

        # self.enhance1 = CrossAttention(in_channels=dim, emb_dim=dim)
        # self.enhance2 = CrossAttention(in_channels=dim, emb_dim=dim)
        # self.enhance3 = CrossAttention(in_channels=dim, emb_dim=dim)

        self.enhance1 = CAMEhance(dim, ratio=8)
        self.enhance2 = CAMEhance(dim, ratio=8)
        self.enhance3 = CAMEhance(dim, ratio=8)


    def forward(self, x):
        batch_size, _, _, _ = x.shape
        attn1 = self.conv0(x)
        attn2 = self.conv1(attn1)
        attn3 = self.conv2(attn2)
        gl = self.NL(x)

        # 1
        # attn1 = attn1 + gl
        # attn2 = attn2 + gl
        # attn3 = attn3 + gl

        # 2
        attn1 = self.enhance1(attn1, gl)
        attn2 = self.enhance2(attn2, gl)
        attn3 = self.enhance3(attn3, gl)

        output = [attn1, attn2, attn3]

        attn = torch.cat([attn1, attn2, attn3], dim=1) # 可以直接相加
        avg_attn = self.avg_pool(attn)
        max_attn = self.max_pool(attn)  # 4 1024 1
        agg = torch.cat([avg_attn, max_attn], dim=1)  # 4 2048
        at = self.fc1(agg)
        at = self.fc2(at)

        a_b = at.reshape(batch_size, 3, self.dim, -1)  # 调整形状，变为 3个全连接层的值
        a_b = self.softmax(a_b)  # 使得两个全连接层对应位置进行softmax
        a_b = list(a_b.chunk(3, dim=1))  # split to a and b   chunk为pytorch方法，将tensor按照指定维度切分成 几个tensor块
        a_b=list(map(lambda x:x.reshape(batch_size,256,1,1),a_b)) # 将所有分块  调整形状，即扩展两维
        V = list(map(lambda x, y: x * y, output, a_b))  # 权重与对应  不同卷积核输出的U 逐元素相乘
        V = reduce(lambda x, y: x + y, V)  # 两个加权后的特征 逐元素相加

        return V

class LargeConvAdd(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.conv0 = nn.Conv2d(dim, dim, 3, padding=1, groups=dim, dilation=1) # 3 9 19
        self.conv1 = nn.Conv2d(dim, dim, 3, padding=3, groups=dim, dilation=3)
        self.conv2 = nn.Conv2d(dim, dim, 3, padding=5, groups=dim, dilation=5)
        self.NL = NonLocalBlock(dim)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1=nn.Sequential(nn.Conv2d(dim * 2, dim // 4,1,bias=False),
                               nn.ReLU(inplace=True))   # 降维
        self.fc2 = nn.Conv2d(dim // 4, dim * 3, kernel_size=1, bias=False)

        self.softmax = nn.Softmax(dim=1)  # 指定dim=1  使得两个全连接层对应位置进行softmax,保证 对应位置a+b+..=1

        self.enhance1 = CAMEhance(dim, ratio=8)
        self.enhance2 = CAMEhance(dim, ratio=8)
        self.enhance3 = CAMEhance(dim, ratio=8)


    def forward(self, x):
        batch_size, _, _, _ = x.shape
        attn1 = self.conv0(x)
        attn2 = self.conv1(attn1)
        attn3 = self.conv2(attn2)
        gl = self.NL(x)

        attn1 = self.enhance1(attn1, gl)
        attn2 = self.enhance2(attn2, gl)
        attn3 = self.enhance3(attn3, gl)

        output = [attn1, attn2, attn3]

        attn =attn1 + attn2 + attn3
        avg_attn = self.avg_pool(attn)
        max_attn = self.max_pool(attn)  # 4 1024 1
        agg = torch.cat([avg_attn, max_attn], dim=1)  # 4 2048
        at = self.fc1(agg)
        at = self.fc2(at)

        a_b = at.reshape(batch_size, 3, self.dim, -1)  # 调整形状，变为 3个全连接层的值
        a_b = self.softmax(a_b)  # 使得两个全连接层对应位置进行softmax
        a_b = list(a_b.chunk(3, dim=1))  # split to a and b   chunk为pytorch方法，将tensor按照指定维度切分成 几个tensor块
        a_b=list(map(lambda x:x.reshape(batch_size,256,1,1),a_b)) # 将所有分块  调整形状，即扩展两维
        V = list(map(lambda x, y: x * y, output, a_b))  # 权重与对应  不同卷积核输出的U 逐元素相乘
        V = reduce(lambda x, y: x + y, V)  # 两个加权后的特征 逐元素相加

        return V

class LargeConvSpatial(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.conv0 = nn.Conv2d(dim, dim, 3, padding=1, groups=dim, dilation=1) # 3 9 19
        self.conv1 = nn.Conv2d(dim, dim, 3, padding=3, groups=dim, dilation=3)
        self.conv2 = nn.Conv2d(dim, dim, 3, padding=5, groups=dim, dilation=5)
        self.NL = NonLocalBlock(dim)

        self.conv_squeeze = nn.Conv2d(2, 3, kernel_size=7, padding=3)
        self.conv = nn.Conv2d(dim, dim, 1)
        # self.enhance1 = CrossAttention(in_channels=dim, emb_dim=dim)
        # self.enhance2 = CrossAttention(in_channels=dim, emb_dim=dim)
        # self.enhance3 = CrossAttention(in_channels=dim, emb_dim=dim)

        self.enhance1 = CAMEhance(dim, ratio=8)
        self.enhance2 = CAMEhance(dim, ratio=8)
        self.enhance3 = CAMEhance(dim, ratio=8)


    def forward(self, x):
        batch_size, _, _, _ = x.shape
        attn1 = self.conv0(x)
        attn2 = self.conv1(attn1)
        attn3 = self.conv2(attn2)
        gl = self.NL(x)

        # 2
        attn1 = self.enhance1(attn1, gl)
        attn2 = self.enhance2(attn2, gl)
        attn3 = self.enhance3(attn3, gl)

        attn = torch.cat([attn1, attn2, attn3], dim=1) # 可以直接相加
        avg_attn = torch.mean(attn, dim=1, keepdim=True)
        max_attn, _ = torch.max(attn, dim=1, keepdim=True)  # 4 1024 1
        agg = torch.cat([avg_attn, max_attn], dim=1)  # 4 2048
        sig = self.conv_squeeze(agg).sigmoid()
        attn = attn1 * sig[:, 0, :, :].unsqueeze(1) + attn2 * sig[:, 1, :, :].unsqueeze(1) + attn3 * sig[:, 2, :, :].unsqueeze(1)
        attn = self.conv(attn)

        return x * attn

if __name__ == "__main__":
    # 示例用法
    B, C, H, W = 1, 256, 64, 64  # 示例批次大小、通道数、高度和宽度
    x = torch.randn(B, C, H, W)
    import time
    start = time.time()
    model = LargeConvAdd(256)
    result = model(x)
    print(time.time() - start)

    a = 1
