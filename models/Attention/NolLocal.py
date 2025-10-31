import torch
import torch.nn as nn
import torchvision


class NonLocalBlock(nn.Module):
    def __init__(self, channel):
        super(NonLocalBlock, self).__init__()
        self.scale = 2
        self.inter_channel = channel // self.scale
        self.conv_phi = nn.Conv2d(in_channels=channel, out_channels=self.inter_channel, kernel_size=1, stride=1,padding=0, bias=False)
        self.conv_theta = nn.Conv2d(in_channels=channel, out_channels=self.inter_channel, kernel_size=1, stride=1, padding=0, bias=False)
        self.conv_g = nn.Conv2d(in_channels=channel, out_channels=self.inter_channel, kernel_size=1, stride=1, padding=0, bias=False)
        self.softmax = nn.Softmax(dim=1)
        self.conv_mask = nn.Conv2d(in_channels=self.inter_channel, out_channels=channel, kernel_size=1, stride=1, padding=0, bias=False)

    def forward(self, x):
        # [N, C, H , W]
        b, c, h, w = x.size()
        # [N, C/2, H * W]
        x_phi = self.conv_phi(x).view(b, self.inter_channel, -1)
        # [N, H * W, C/2]
        x_theta = self.conv_theta(x).view(b, self.inter_channel, -1).permute(0, 2, 1).contiguous()
        x_g = self.conv_g(x).view(b, self.inter_channel, -1).permute(0, 2, 1).contiguous()
        # [N, H * W, H * W]
        mul_theta_phi = torch.matmul(x_theta, x_phi)
        mul_theta_phi = self.softmax(mul_theta_phi)
        # [N, H * W, C/2]
        mul_theta_phi_g = torch.matmul(mul_theta_phi, x_g)
        # [N, C/2, H, W]
        mul_theta_phi_g = mul_theta_phi_g.permute(0,2,1).contiguous().view(b,self.inter_channel, h, w)
        # [N, C, H , W]
        mask = self.conv_mask(mul_theta_phi_g)
        out = mask + x
        return out


if __name__=='__main__':
    model = NonLocalBlock(channel=512)

    import time
    start = time.time()
    input = torch.randn(1, 512, 64, 64)
    out = model(input)
    print(time.time() -start)
    # 计算参数数量
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Total number of parameters: {num_params}")

    from thop import profile
    from thop import clever_format

    # 计算 FLOPs 和参数量
    flops, params = profile(model, inputs=(input,))

    # 格式化输出，使其更易读
    flops, params = clever_format([flops, params], "%.3f")

    print(f"FLOPs: {flops}")
    print(f"Params: {params}")