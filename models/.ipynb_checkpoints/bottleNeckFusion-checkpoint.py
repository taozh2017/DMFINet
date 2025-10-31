import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from matplotlib import pyplot as plt

from data.EN2CTDataloader import get_loader
from network import ResnetBlock

# number one
class CrossScaleAttentionFusion(nn.Module):
    def __init__(self, in_channels1, in_channels2):
        super(CrossScaleAttentionFusion, self).__init__()
        self.in_channels1 = in_channels1
        self.in_channels2 = in_channels2

        # 注意力机制
        self.attention1 = nn.Sequential(
            nn.Conv2d(in_channels1 + in_channels2, in_channels1, kernel_size=1),
            nn.BatchNorm2d(in_channels1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels1, in_channels1, kernel_size=1),
            nn.Sigmoid()
        )

        self.attention2 = nn.Sequential(
            nn.Conv2d(in_channels1 + in_channels2, in_channels2, kernel_size=1),
            nn.BatchNorm2d(in_channels2),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels2, in_channels2, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, data1, data2):
        # data1 shape: (batch_size, in_channels1, 100, 100)
        # data2 shape: (batch_size, in_channels2, 50, 50)

        # 上采样 data2 到 data1 的尺寸
        upsampled_data2 = F.interpolate(data2, size=(data1.size(2), data1.size(3)), mode='bilinear',
                                        align_corners=False)

        # 下采样 data1 到 data2 的尺寸
        downsampled_data1 = F.interpolate(data1, size=(data2.size(2), data2.size(3)), mode='bilinear',
                                          align_corners=False)

        # 计算注意力权重
        combined1 = torch.cat((data1, upsampled_data2), dim=1)
        attention1 = self.attention1(combined1)

        combined2 = torch.cat((downsampled_data1, data2), dim=1)
        attention2 = self.attention2(combined2)

        # 融合特征
        fused_data1 = data1 * attention1 + upsampled_data2 * (1 - attention1)
        fused_data2 = data2 * attention2 + downsampled_data1 * (1 - attention2)

        return fused_data1, fused_data2



