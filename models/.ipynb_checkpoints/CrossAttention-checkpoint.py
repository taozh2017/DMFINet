import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class CrossAttention(nn.Module):
    def __init__(self, in_channels, emb_dim, att_dropout=0.0, dropout=0.0):
        super(CrossAttention, self).__init__()
        self.emb_dim = emb_dim
        self.scale = emb_dim ** -0.5

        self.proj_in1 = nn.Conv2d(in_channels, emb_dim, kernel_size=1, stride=1, padding=0)
        self.proj_in2 = nn.Conv2d(in_channels, emb_dim, kernel_size=1, stride=1, padding=0)

        self.Wq = nn.Linear(emb_dim, emb_dim)
        self.Wk = nn.Linear(emb_dim, emb_dim)
        self.Wv = nn.Linear(emb_dim, emb_dim)

        self.proj_out = nn.Conv2d(emb_dim, in_channels, kernel_size=1, stride=1, padding=0)

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

        # 将x和context展开
        x = rearrange(x, 'b c h w -> b (h w) c')  # [batch_size, h*w, emb_dim]
        context = rearrange(context, 'b c h w -> b (h w) c')  # [batch_size, (h/2)*(w/2), emb_dim]

        # 生成查询（Q）、键（K）和值（V）
        Q = self.Wq(x)  # [batch_size, h*w, emb_dim]
        K = self.Wk(context)  # [batch_size, (h/2)*(w/2), emb_dim]
        V = self.Wv(context)

        # 计算注意力权重
        att_weights = torch.einsum('bid,bjd -> bij', Q, K)  # [batch_size, h*w, (h/2)*(w/2)]
        att_weights = att_weights * self.scale

        # 如果提供了 pad_mask，则屏蔽特定位置的权重
        if pad_mask is not None:
            att_weights = att_weights.masked_fill(pad_mask, -1e9)

        # 计算 softmax 注意力权重
        att_weights = F.softmax(att_weights, dim=-1)

        # 使用注意力权重对 V 进行加权求和
        out = torch.einsum('bij, bjd -> bid', att_weights, V)  # [batch_size, h*w, emb_dim]

        # 将输出特征重新排列回 [batch_size, c, h, w] 形状
        out = rearrange(out, 'b (h w) c -> b c h w', h=h, w=w)  # [batch_size, emb_dim, h, w]

        # 将输出特征重新投影回输入的通道数
        out = self.proj_out(out)  # [batch_size, c, h, w]

        return out, att_weights


# 示例使用
if __name__ == '__main__':
    cross_attention = CrossAttention(in_channels=64, emb_dim=512)

    # 生成随机输入数据
    x = torch.randn(1, 64, 64, 64)  # [batch_size, c, h, w]
    context = torch.randn(3, 64, 32, 32)  # [batch_size, c, h/2, w/2]

    out, att_weights = cross_attention(x, context)
    print(out.shape)  # 输出的形状应为 [3, 512, 32, 32]
