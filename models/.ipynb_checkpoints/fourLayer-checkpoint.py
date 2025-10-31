import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from matplotlib import pyplot as plt
from data.EN2CTDataloader import get_loader

from models.Attention.MultiScaleCrossAttention import CrossAttentionBlock
from models.Attention.CrossAttention import CrossAttention
# from .bottleNeckFusion import CrossScaleAttentionFusion
from network import ResnetBlock
from util.networkUtil import *
import functools

class AttentionGate(nn.Module):
    def __init__(self, F_g, F_l, F_int, kernel_size=3):
        super(AttentionGate, self).__init__()

        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=kernel_size, stride=1, padding=kernel_size // 2, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=kernel_size, stride=1, padding=kernel_size // 2, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.activation = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.activation(g1 + x1)
        psi = self.psi(psi)

        return x * psi

class ResidualBlock(nn.Module):
    def __init__(self, in_chan, out_chan, act, norm_layer, use_bias):
        super(ResidualBlock, self).__init__()
        self.conv1 = LUConv(in_chan, out_chan, act, norm_layer, use_bias)
        self.conv2 = nn.Conv2d(out_chan, out_chan, kernel_size=3, padding=1, bias=use_bias)
        self.bn2 = norm_layer(out_chan)
        self.shortcut = nn.Sequential()
        if in_chan != out_chan:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_chan, out_chan, kernel_size=1, stride=1, bias=use_bias),
                norm_layer(out_chan)
            )

        if act == 'relu':
            self.activation = nn.ReLU(out_chan)
        elif act == 'prelu':
            self.activation = nn.PReLU(out_chan)
        elif act == 'elu':
            self.activation = nn.ELU(inplace=True)
        else:
            raise

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.bn2(out)
        out += self.shortcut(x)
        return self.activation(out)

def _make_nConv(in_channel, out_cha, act, res_net=False, norm_layer=nn.BatchNorm2d, use_bias=False):
    if res_net:
        return ResidualBlock(in_channel, out_cha, act, norm_layer, use_bias)
    else:
        layer1 = LUConv(in_channel, out_cha, act, norm_layer, use_bias)
        layer2 = LUConv(out_cha, out_cha, act, norm_layer, use_bias)
        return nn.Sequential(layer1, layer2)

class DownTransition(nn.Module):
    def __init__(self, in_channel, out_cha, act, res_net = False, norm_layer=nn.BatchNorm2d, use_bias=False):
        super(DownTransition, self).__init__()
        self.ops = _make_nConv(in_channel, out_cha, act, res_net, norm_layer, use_bias)
        self.maxpool = nn.MaxPool2d(2)

    def forward(self, x):
        out_before_pool = self.ops(x)
        out = self.maxpool(out_before_pool)

        return out, out_before_pool

class LUConv(nn.Module):
    def __init__(self, in_chan, out_chan, act, norm_layer, use_bias):
        super(LUConv, self).__init__()
        self.conv1 = nn.Conv2d(in_chan, out_chan, kernel_size=3, padding=1, bias=use_bias)
        self.bn1 = norm_layer(out_chan)

        if act == 'relu':
            self.activation = nn.ReLU(out_chan)
        elif act == 'prelu':
            self.activation = nn.PReLU(out_chan)
        elif act == 'elu':
            self.activation = nn.ELU(inplace=True)
        else:
            raise

    def forward(self, x):
        out = self.activation(self.bn1(self.conv1(x)))
        return out

class DownsampeModel(nn.Module):
    def __init__(self):
        super(DownsampeModel, self).__init__()
        # 定义转置卷积层等操作

    def forward(self, x):
        downsampled_x = F.max_pool2d(x, kernel_size=(2, 2), stride=(2, 2))

        return downsampled_x

class UpLayer(nn.Module):
    def __init__(self, in_cha, out_cha, res_net= False, norm_layer = nn.BatchNorm2d, use_bias=False):
        super(UpLayer, self).__init__()
        # 在这里可以定义需要初始化的层，如反卷积层等
        self.model = nn.Sequential(nn.ConvTranspose2d(in_cha, out_cha,
                                         kernel_size=3, stride=2,
                                         padding=1, output_padding=1,
                                         bias=use_bias),
                                  norm_layer(out_cha),
                                  nn.ReLU(True))
        self.layer = _make_nConv(out_cha, out_cha, "relu", res_net, norm_layer, use_bias=use_bias)

    def forward(self, x):
        # 在这里定义层的操作
        # 例如，使用反卷积层进行上采样操作
        x = self.model(x)
        x = self.layer(x)
        return x

class mainFusion1(nn.Module):
    def __init__(self, in_cha, out_cha, norm_layer = nn.BatchNorm2d):
        super(mainFusion1, self).__init__()
        # 在这里可以定义需要初始化的层，如反卷积层等
        self.up = nn.Sequential(nn.ConvTranspose2d(in_cha * 2, in_cha,
                                         kernel_size=3, stride=2,
                                         padding=1, output_padding=1,
                                         bias=True),
                                  norm_layer(in_cha),
                                  nn.ReLU(True))


        self.conv1 = LUConv(out_cha * 4, out_cha, "relu")

    def forward(self, main_skip, up_skip, main_decoder):  # 分别是主要分支对应得skipconnectin， 上面分支对应的skip, 主分支下层特征
        x = torch.cat([up_skip, main_decoder], dim=1)
        x_up = self.up(x)
        xy = torch.cat([main_skip, x_up], dim=1)
        xy = self.conv1(xy)

        return xy

class mainFusionSharedParemeter(nn.Module):
    def __init__(self, in_cha, out_cha, norm_layer=nn.BatchNorm2d):
        super(mainFusionSharedParemeter, self).__init__()
        # 在这里可以定义需要初始化的层，如反卷积层等
        self.conv1 = nn.Sequential(nn.Conv2d(in_cha * 2, in_cha, kernel_size=1, stride=1, padding=0),
                                   norm_layer(in_cha),
                                   nn.ReLU(True))
        self.up1 = nn.Sequential(nn.ConvTranspose2d(in_cha, in_cha,
                                                   kernel_size=3, stride=2,
                                                   padding=1, output_padding=1,
                                                   bias=True),
                                norm_layer(in_cha),
                                nn.ReLU(True))
        self.att = AttentionGate(in_cha, in_cha, int(in_cha / 2))

        self.conv2 = LUConv(in_cha, out_cha, "relu")

    def forward(self, main_skip, up_skip, main_decoder):  # 分别是主要分支对应得skipconnectin， 上面分支对应的skip, 主分支下层特征
        x = torch.cat([up_skip, main_decoder], dim=1) # 128 64 64
        x = self.conv1(x)  # 64 64 64
        x = self.up1(x) # 64 128 128

        y = self.up1(up_skip) # 64 128 128
        y = torch.cat([y, main_skip], dim=1) # 128 128 128
        y = self.conv1(y) # 64 128 128

        y_skip = self.att(x, y)
        xy = y_skip + x
        xy = self.conv2(xy)
        return xy

class mainFusion2(nn.Module):
    def __init__(self, in_cha, out_cha, res_net=False, norm_layer=nn.BatchNorm2d, use_bias=False):
        super(mainFusion, self).__init__()
        # 在这里可以定义需要初始化的层，如反卷积层等
        self.conv1 = nn.Sequential(nn.Conv2d(in_cha * 2, in_cha, kernel_size=1, stride=1, padding=0, bias=use_bias),
                                   norm_layer(in_cha),
                                   nn.ReLU(True))
        self.conv2 = nn.Sequential(nn.Conv2d(in_cha * 2, in_cha, kernel_size=1, stride=1, padding=0, bias=use_bias),
                                   norm_layer(in_cha),
                                   nn.ReLU(True))
        self.up1 = nn.Sequential(nn.ConvTranspose2d(in_cha, in_cha,
                                                   kernel_size=3, stride=2,
                                                   padding=1, output_padding=1,
                                                   bias=use_bias),
                                norm_layer(in_cha),
                                nn.ReLU(True))
        self.up2 = nn.Sequential(nn.ConvTranspose2d(in_cha, in_cha,
                                                    kernel_size=3, stride=2,
                                                    padding=1, output_padding=1,
                                                    bias=use_bias),
                                 norm_layer(in_cha),
                                 nn.ReLU(True))
        self.att = AttentionGate(in_cha, in_cha, int(in_cha / 2))

        self.conv3 = _make_nConv(in_cha, out_cha, "relu", res_net, norm_layer, use_bias=use_bias)

    def forward(self, main_skip, up_skip, main_decoder):  # 分别是主要分支对应得skipconnectin， 上面分支对应的skip, 主分支下层特征
        x = torch.cat([up_skip, main_decoder], dim=1) # 128 64 64
        x = self.conv1(x)  # 64 64 64
        x = self.up1(x) # 64 128 128

        y = self.up2(up_skip) # 64 128 128
        y = torch.cat([y, main_skip], dim=1) # 128 128 128
        y = self.conv2(y) # 64 128 128

        y_skip = self.att(x, y)
        xy = y_skip + x
        xy = self.conv3(xy)
        return xy

class mainFusion(nn.Module):
    def __init__(self, in_cha, out_cha, res_net=False, norm_layer=nn.BatchNorm2d, use_bias=False):
        super(mainFusion, self).__init__()
        # 在这里可以定义需要初始化的层，如反卷积层等
        self.conv1 = nn.Sequential(nn.Conv2d(in_cha * 2, in_cha, kernel_size=1, stride=1, padding=0, bias=use_bias),
                                   norm_layer(in_cha),
                                   nn.ReLU(True))

        self.up1 = nn.Sequential(nn.ConvTranspose2d(in_cha, in_cha,
                                                   kernel_size=3, stride=2,
                                                   padding=1, output_padding=1,
                                                   bias=use_bias),
                                norm_layer(in_cha),
                                nn.ReLU(True))
        self.up2 = nn.Sequential(nn.ConvTranspose2d(in_cha, in_cha,
                                                    kernel_size=3, stride=2,
                                                    padding=1, output_padding=1,
                                                    bias=use_bias),
                                 norm_layer(in_cha),
                                 nn.ReLU(True))
        self.att = AttentionGate(in_cha, in_cha, int(in_cha / 2))

        self.conv2 = _make_nConv(in_cha, out_cha, "relu", res_net, norm_layer, use_bias=use_bias)

    def forward(self, main_skip, up_skip, main_decoder):  # 分别是主要分支对应得skipconnectin， 上面分支对应的skip, 主分支下层特征
        x = self.up1(main_decoder) # 64 128 128

        y = self.up2(up_skip) # 64 128 128
        y = torch.cat([y, main_skip], dim=1) # 128 128 128
        y = self.conv1(y) # 64 128 128

        y_skip = self.att(x, y)
        xy = y_skip + x # 64 128 128
        xy = self.conv2(xy)
        return xy

def maskRand(img, patch_size, mask_ratio):
    img = patchify(img, patch_size)  #1 1024 256
    # masking: length -> length * mask_ratio
    img = (img + 1) / 2  # -1 1  -> 0 1
    x, mask, ids_restore, mask_image, mask_expand = random_masking(img, mask_ratio)
    mask_image = (mask_image * 2) - 1  # 0 1 -> -1 1
    mask_image = unpatchify(mask_image, patch_size=patch_size)

    return x, mask, ids_restore, mask_image, mask_expand


class MyModel(nn.Module):
    def __init__(self, act='relu', base_filter = 32, bottleneck_layers = 3, re_outcha = 1,
                 patch_size = 8, mask_ratio = 0.2, res_net = False, norm_layer=nn.BatchNorm2d, att_dim=128):
        super(MyModel, self).__init__()
        self.patch_size = patch_size
        self.mask_ratio = mask_ratio
        self.base_filter = base_filter
        self.att_dim = att_dim
        print("res_net value :  ", res_net)
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        print("using bias : ", use_bias)
        print("using layer four")

        # 权重共享
        self.down_one = DownTransition(1, self.base_filter, act, res_net=res_net, norm_layer=norm_layer, use_bias=use_bias)
        self.down_two = DownTransition(self.base_filter, self.base_filter * 2, act, res_net=res_net, norm_layer=norm_layer, use_bias=use_bias)
        self.down_three = DownTransition(self.base_filter * 2, self.base_filter * 4, act, res_net=res_net, norm_layer=norm_layer, use_bias=use_bias)
        self.down_four = DownTransition(self.base_filter * 4, self.base_filter * 8, act, res_net=res_net, norm_layer=norm_layer, use_bias=use_bias)
        self.bottle_neck_channel = self.base_filter * 8

        # 重建模块
        # 第一次的底层，两者没有交互。
        # self.re_bottlneck = []
        # for _ in range(bottleneck_layers):
        #     self.re_bottlneck.append(_make_nConv(self.bottle_neck_channel, self.bottle_neck_channel, act, res_net=res_net, norm_layer=norm_layer, use_bias=use_bias))
        #     # self.re_bottlneck.append(ResnetBlock(self.base_filter * 4, padding_type="'reflect'", norm_layer=norm_layer, use_dropout=False, use_bias=True))
        # self.re_bottlneck_model = nn.Sequential(*self.re_bottlneck)

        # 第二次底层，尝试交互
        self.re_bottleNeckOne = _make_nConv(self.bottle_neck_channel, self.bottle_neck_channel, act, res_net=res_net, norm_layer=norm_layer, use_bias=use_bias)
        self.re_bottleNeckTwo = _make_nConv(self.bottle_neck_channel, self.bottle_neck_channel, act, res_net=res_net, norm_layer=norm_layer, use_bias=use_bias)
        self.re_bottleNeckThree = _make_nConv(self.bottle_neck_channel, self.bottle_neck_channel, act, res_net=res_net, norm_layer=norm_layer, use_bias=use_bias)

        # 上面特征提取模块 参数不共享
        self.up_down_one = DownTransition(1, self.base_filter, act, res_net=res_net, norm_layer=norm_layer, use_bias=use_bias)
        self.up_down_two = DownTransition(self.base_filter, self.base_filter * 2, act, res_net=res_net, norm_layer=norm_layer, use_bias=use_bias)
        self.up_down_three = DownTransition(self.base_filter*2, self.base_filter * 4, act, res_net=res_net, norm_layer=norm_layer, use_bias=use_bias)
        self.up_down_four = DownTransition(self.base_filter*4, self.base_filter * 8, act, res_net=res_net, norm_layer=norm_layer, use_bias=use_bias)

        # # 主分支模块
        # 第一次，没有交互
        # self.main_bottlneck = []
        # for _ in range(bottleneck_layers):
        #     self.main_bottlneck.append(_make_nConv(self.bottle_neck_channel, self.bottle_neck_channel, act, res_net=res_net, norm_layer=norm_layer, use_bias=use_bias))
        #     # self.re_bottlneck.append(ResnetBlock(self.base_filter * 4, padding_type="'reflect'", norm_layer=norm_layer, use_dropout=False, use_bias=True))
        # self.main_bottlneck_model = nn.Sequential(*self.main_bottlneck)

        # 第二次底层，尝试交互
        self.main_bottleNeckOne = _make_nConv(self.bottle_neck_channel, self.bottle_neck_channel, act, res_net=res_net, norm_layer=norm_layer, use_bias=use_bias)
        self.main_bottleNeckTwo = _make_nConv(self.bottle_neck_channel, self.bottle_neck_channel, act, res_net=res_net, norm_layer=norm_layer, use_bias=use_bias)
        self.main_bottleNeckThree = _make_nConv(self.bottle_neck_channel, self.bottle_neck_channel, act, res_net=res_net, norm_layer=norm_layer, use_bias=use_bias)

        # TWO
        self.att_dim = self.base_filter * 8
        self.bottle_fusionOne = CrossAttention(self.bottle_neck_channel, self.att_dim)
        self.bottle_fusionTwo = CrossAttention(self.bottle_neck_channel, self.att_dim)
        self.bottle_fusionThree = CrossAttention(self.bottle_neck_channel, self.att_dim)

        # # Three
        # self.bottle_fusionOne = CrossAttentionBlock(img_size=64, img_size_y=32, patch_size=4, in_chans=self.bottle_neck_channel, embed_dim=2048, depth=3, num_heads=8,
        #         mlp_ratio=4., qkv_bias=True, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
        #          drop_path_rate=0., hybrid_backbone=None, norm_layer=nn.LayerNorm, patch_first=True, un_patch = False)
        # self.bottle_fusionTwo = CrossAttentionBlock(img_size=64, img_size_y=32, patch_size=4, in_chans=self.bottle_neck_channel, embed_dim=2048, depth=3, num_heads=8,
        #         mlp_ratio=4., qkv_bias=True, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
        #          drop_path_rate=0., hybrid_backbone=None, norm_layer=nn.LayerNorm, patch_first=False, un_patch = False)
        # self.bottle_fusionThree = CrossAttentionBlock(img_size=64, img_size_y=32, patch_size=4, in_chans=self.bottle_neck_channel, embed_dim=2048, depth=3, num_heads=8,
        #         mlp_ratio=4., qkv_bias=True, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
        #          drop_path_rate=0., hybrid_backbone=None, norm_layer=nn.LayerNorm, patch_first=False, un_patch = True)
        self.mainFusionFour = mainFusion(self.base_filter * 8, self.base_filter * 4, res_net=res_net, norm_layer=norm_layer, use_bias=use_bias)
        self.mainFusionOne = mainFusion(self.base_filter * 4, self.base_filter * 2, res_net=res_net, norm_layer=norm_layer, use_bias=use_bias)
        self.mainFusionTwo = mainFusion(self.base_filter * 2, self.base_filter, res_net=res_net, norm_layer=norm_layer, use_bias=use_bias)
        self.mainFusionThree = mainFusion(self.base_filter, int((self.base_filter + 1) /2 ), res_net=res_net, norm_layer=norm_layer, use_bias=use_bias)
        self.main_last = nn.Sequential(nn.ReflectionPad2d(1),
                                     nn.Conv2d(int((self.base_filter * 1) / 2), re_outcha, kernel_size=3, padding=0),
                                     nn.Tanh())

        # upsample reconstruction
        self.re_upfour = UpLayer(self.base_filter * 8, self.base_filter * 4, res_net=res_net, norm_layer=norm_layer, use_bias=use_bias)

        self.re_upone = UpLayer(self.base_filter * 4, self.base_filter * 2, res_net=res_net, norm_layer=norm_layer, use_bias=use_bias)
        self.re_uptwo = UpLayer(self.base_filter * 2, self.base_filter * 1, res_net=res_net, norm_layer=norm_layer, use_bias=use_bias)
        self.re_upthree = UpLayer(self.base_filter * 1, int((self.base_filter * 1)/2), res_net=res_net, norm_layer=norm_layer, use_bias=use_bias)
        self.re_last = nn.Sequential(nn.ReflectionPad2d(1),
                                     nn.Conv2d(int((self.base_filter * 1)/2), re_outcha, kernel_size=3, padding=0),
                                     nn.Tanh())
        self.L1Loss = nn.L1Loss()
        self.noise = NoiseLayer(noise_std=0.01)

    def forward(self, x, down_image, re_down_image):
        # process the main branch
        self.out_one, self.skip_outone = self.down_one(x)
        self.out_two, self.skip_outtwo = self.down_two(self.out_one)
        self.out_three, self.skip_outthree = self.down_three(self.out_two) # 128 64 64
        self.out_four, self.skip_outfour = self.down_four(self.out_three) # 256 32 32

        #process the reconstruct branch
        x_masked, mask, ids_restore, mask_image, mask_expand = maskRand(img=re_down_image, patch_size=self.patch_size, mask_ratio=self.mask_ratio)

        re_one, re_skipone = self.down_one(mask_image)
        re_two, re_skiptwo = self.down_two(re_one)
        re_three, re_skipthree = self.down_three(re_two) # 128 32 32
        re_four, re_skipfour = self.down_four(re_three) # 256 16 16

        # process bottleneck fusion
        # TWO
        self.main_bottle = self.out_four
        self.re_bottle = re_four

        self.main_bottle = self.main_bottleNeckOne(self.main_bottle)
        self.re_bottle = self.re_bottleNeckOne(self.re_bottle)
        self.main_bottle, self.weights = self.bottle_fusionOne(self.main_bottle, self.re_bottle)
        self.main_bottle = self.main_bottleNeckTwo(self.main_bottle)
        self.re_bottle = self.re_bottleNeckTwo(self.re_bottle)
        self.main_bottle, self.weights = self.bottle_fusionTwo(self.main_bottle, self.re_bottle)
        self.main_bottle = self.main_bottleNeckThree(self.main_bottle)
        self.re_bottle = self.re_bottleNeckThree(self.re_bottle)
        self.main_bottle, self.weights = self.bottle_fusionThree(self.main_bottle, self.re_bottle)

        # three
        # self.main_bottle = self.out_three
        # self.re_bottle = re_three
        # self.main_bottle = self.bottle_fusionOne(self.main_bottle, self.re_bottle)
        # self.re_bottle = self.re_bottleNeckOne(re_three)
        # self.main_bottle = self.bottle_fusionTwo(self.main_bottle, self.re_bottle)
        # self.re_bottle = self.re_bottleNeckTwo(self.re_bottle)
        # self.main_bottle = self.bottle_fusionThree(self.main_bottle, self.re_bottle)
        # self.re_bottle = self.re_bottleNeckThree(self.re_bottle)

        # process the reconstruction
        # up = self.re_bottlneck_model(re_three)
        up = self.re_bottle

        up = self.re_upfour(up)
        up = self.re_upone(up)
        up = self.re_uptwo(up)
        up = self.re_upthree(up)
        re_result = self.re_last(up)

        # process the up branch
        up_one, up_skipone = self.up_down_one(down_image) # 1 32 128 128    1 32 256 256
        up_two, up_skiptwo = self.up_down_two(up_one)
        up_three, up_skipthree = self.up_down_three(up_two)
        up_four, up_skipfour = self.up_down_four(up_three)
        # 可以在主分支下采样时将上面分支的特征进行融合
        # 将上分支下采样的结果和主分支的skip进行融合

        # process the main up branch
#         self.out_three = self.noise(self.out_three)

        # main_up = self.main_bottlneck_model(self.out_three)
        main_up = self.main_bottle

        main_up_four = self.mainFusionFour(self.skip_outfour, up_skipfour, main_up)
        main_up_one = self.mainFusionOne(self.skip_outthree, up_skipthree, main_up_four)
        main_up_two = self.mainFusionTwo(self.skip_outtwo, up_skiptwo, main_up_one)
        main_up_three = self.mainFusionThree(self.skip_outone, up_skipone, main_up_two)
        main_result = self.main_last(main_up_three)

        return mask_image, re_result, main_result


if __name__ == '__main__':
    model = MyModel(act="relu", base_filter=32, mask_ratio=0.2, patch_size=8, res_net=True, norm_layer=nn.BatchNorm2d, att_dim=256)
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    a = torch.rand((1, 1, 512, 512))
    a[:, :, :200, :] = 1.0
    a[:, :, 200:400, :] = 0.5
    a[:, :, 400:, :] = 0
    data_loader = get_loader(batchsize=1, shuffle=False, pin_memory=True, source_modal='en', target_modal='ct',
               img_size=512, img_root="../data/", model="train/", data_rate=1, num_workers=24, sort=False, argument=True,
               random=False, re_augment = True, re_down = True)


    def count_parameters(model):
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Total Parameters: {total_params}")

        # for name, param in model.named_parameters():
        #     if param.requires_grad:
        #         print(f"{name}: {param.numel()}")

    count_parameters(model)


    for i, data in enumerate(data_loader):
        print("epoch : ", i)
        img = data['A'].float()
        down_image = data['A_down_image'].float()
        re_down_image = data['A_re_down_image'].float()
        start_time = time.time()
        mask_image, re_result, main_result = model(a, down_image, re_down_image)
        end_time = time.time()
        print("using time: ", end_time - start_time)
        if i % 1 == 0:
            # print(re_loss)
            # 可视化原始图像和处理后的图像
            plt.figure(figsize=(12, 6))
            plt.subplot(131)
            plt.imshow(img[0][0], cmap='gray')  # 显示灰度图像
            plt.title("Original Image")
            plt.subplot(132)
            plt.imshow(mask_image[0][0].detach(), cmap='gray')  # 显示处理后的灰度图像
            plt.title("Processed Image")
            plt.subplot(133)
            plt.imshow(re_result[0][0].detach(), cmap='gray')  # 显示处理后的灰度图像
            plt.title("Processed Image")
            plt.show()