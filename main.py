import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from matplotlib import pyplot as plt
from network import ResnetBlock
from util import *

def _make_nConv(in_channel, out_cha, act):  # 可以使用ResNet
    layer1 = LUConv(in_channel, out_cha, act)
    layer2 = LUConv(out_cha, out_cha, act)

    return nn.Sequential(layer1, layer2)

class DownTransition(nn.Module):
    def __init__(self, in_channel, out_cha, act):
        super(DownTransition, self).__init__()
        self.ops = _make_nConv(in_channel, out_cha, act)
        self.maxpool = nn.MaxPool2d(2)

    def forward(self, x):
        out_before_pool = self.ops(x)
        out = self.maxpool(out_before_pool)

        return out, out_before_pool

class LUConv(nn.Module):
    def __init__(self, in_chan, out_chan, act):
        super(LUConv, self).__init__()
        self.conv1 = nn.Conv2d(in_chan, out_chan, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_chan)

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
    def __init__(self, in_cha, out_cha, norm_layer = nn.BatchNorm2d):
        super(UpLayer, self).__init__()
        # 在这里可以定义需要初始化的层，如反卷积层等
        self.model = nn.Sequential(nn.ConvTranspose2d(in_cha, out_cha,
                                         kernel_size=3, stride=2,
                                         padding=1, output_padding=1,
                                         bias=True),
                                  norm_layer(out_cha),
                                  nn.ReLU(True))
        self.layer = LUConv(out_cha, out_cha, "relu")

    def forward(self, x):
        # 在这里定义层的操作
        # 例如，使用反卷积层进行上采样操作
        x = self.model(x)
        x = self.layer(x)
        return x

class mainFusion(nn.Module):
    def __init__(self, in_cha, out_cha, norm_layer = nn.BatchNorm2d):
        super(mainFusion, self).__init__()
        # 在这里可以定义需要初始化的层，如反卷积层等
        self.up = nn.Sequential(nn.ConvTranspose2d(in_cha * 2, in_cha,
                                         kernel_size=3, stride=2,
                                         padding=1, output_padding=1,
                                         bias=True),
                                  norm_layer(in_cha),
                                  nn.ReLU(True))


        self.conv1 = LUConv(out_cha * 4, out_cha, "relu")

    def forward(self, main_skip, down_skip, down_feature):  # 分别是主要分支对应得skipconnectin， 上面分支对应的skip, 主分支下层特征
        x = torch.cat([down_skip, down_feature], dim=1)
        x_up = self.up(x)
        xy = torch.cat([main_skip, x_up], dim=1)
        xy = self.conv1(xy)

        return xy


def maskRand(img, patch_size, mask_ratio):
    img = patchify(img, patch_size)  #1 1024 256
    # masking: length -> length * mask_ratio
    x, mask, ids_restore, mask_image, mask_expand = random_masking(img, mask_ratio)

    return x, mask, ids_restore, mask_image, mask_expand


class MyModel(nn.Module):
    def __init__(self, act='relu', base_filter = 32, bottleneck_layers = 3, norm_layer=nn.BatchNorm2d, re_outcha = 1,
                 patch_size = 8, mask_ratio = 0.2):
        super(MyModel, self).__init__()
        self.patch_size = patch_size
        self.mask_ratio = mask_ratio
        self.base_filter = base_filter

        self.down_first = DownsampeModel()
        # 权重共享
        self.down_one = DownTransition(1, self.base_filter, act)
        self.down_two = DownTransition(self.base_filter, self.base_filter * 2, act)
        self.down_three = DownTransition(self.base_filter*2, self.base_filter * 4, act)
        self.bottle_neck_channel = self.base_filter * 4


        # 重建模块
        self.re_bottlneck = []
        for _ in range(bottleneck_layers):
            self.re_bottlneck.append(_make_nConv(self.bottle_neck_channel, self.bottle_neck_channel, act))
            # self.re_bottlneck.append(ResnetBlock(self.base_filter * 4, padding_type="'reflect'", norm_layer=norm_layer, use_dropout=False, use_bias=True))
        self.re_bottlneck_model = nn.Sequential(*self.re_bottlneck)
        # upsample
        self.re_upone = UpLayer(self.base_filter * 4, self.base_filter * 2)
        self.re_uptwo = UpLayer(self.base_filter * 2, self.base_filter * 1)
        self.re_upthree = UpLayer(self.base_filter * 1, int((self.base_filter * 1)/2))
        self.re_last = nn.Sequential(nn.ReflectionPad2d(1),
                                     nn.Conv2d(int((self.base_filter * 1)/2), re_outcha, kernel_size=3, padding=0),
                                     nn.Tanh())

        # 上面特征提取模块 参数不共享
        self.up_down_one = DownTransition(1, self.base_filter, act)
        self.up_down_two = DownTransition(self.base_filter, self.base_filter * 2, act)
        self.up_down_three = DownTransition(self.base_filter*2, self.base_filter * 4, act)

        # 主分支模块
        self.main_bottlneck = []
        for _ in range(bottleneck_layers):
            self.main_bottlneck.append(_make_nConv(self.bottle_neck_channel, self.bottle_neck_channel, act))
            # self.re_bottlneck.append(ResnetBlock(self.base_filter * 4, padding_type="'reflect'", norm_layer=norm_layer, use_dropout=False, use_bias=True))
        self.main_bottlneck_model = nn.Sequential(*self.main_bottlneck)
        self.mainFusionOne = mainFusion(self.base_filter * 4, self.base_filter * 2)
        self.mainFusionTwo = mainFusion(self.base_filter * 2, self.base_filter)
        self.mainFusionThree = mainFusion(self.base_filter, int((self.base_filter + 1) /2 ))
        self.main_last = nn.Sequential(nn.ReflectionPad2d(1),
                                     nn.Conv2d(int((self.base_filter * 1) / 2), re_outcha, kernel_size=3, padding=0),
                                     nn.Tanh())


        # upsample
        self.re_upone = UpLayer(self.base_filter * 4, self.base_filter * 2)
        self.re_uptwo = UpLayer(self.base_filter * 2, self.base_filter * 1)
        self.re_upthree = UpLayer(self.base_filter * 1, int((self.base_filter * 1) / 2))
        self.re_last = nn.Sequential(nn.ReflectionPad2d(1),
                                     nn.Conv2d(int((self.base_filter * 1) / 2), re_outcha, kernel_size=3, padding=0),
                                     nn.Tanh())


        self.L1Loss = nn.L1Loss()

    def forward(self, x):

        # process the main branch
        self.out_one, self.skip_outone = self.down_one(x)
        self.out_two, self.skip_outtwo = self.down_two(self.out_one)
        self.out_three, self.skip_outthree = self.down_three(self.out_two) # 128 64 64

        #process the reconstruct branch
        # TODO: 给下采样图片添加mask done
        re_down = self.down_first(x)
        x_masked, mask, ids_restore, mask_image, mask_expand = maskRand(img=re_down, patch_size=self.patch_size, mask_ratio=self.mask_ratio)
        mask_image = unpatchify(mask_image, patch_size=self.patch_size)
        re_one, re_skipone = self.down_one(mask_image)
        re_two, re_skiptwo = self.down_two(re_one)
        re_three, re_skipthree = self.down_three(re_two) # 128 32 32
        up = self.re_bottlneck_model(re_three)
        up = self.re_upone(up)
        up = self.re_uptwo(up)
        up = self.re_upthree(up)
        re_result = self.re_last(up)


        # process the up branch
        x2 = self.down_first(x)
        up_one, up_skipone = self.up_down_one(x2) # 1 32 128 128    1 32 256 256
        up_two, up_skiptwo = self.up_down_two(up_one)
        up_three, up_skipthree = self.up_down_three(up_two)
        # 可以在主分支下采样时将上面分支的特征进行融合
        # 将上分支下采样的结果和主分支的skip进行融合

        # process the main up branch
        main_up = self.main_bottlneck_model(self.out_three)
        main_up_one = self.mainFusionOne(self.skip_outthree, up_skipthree, main_up)
        main_up_two = self.mainFusionTwo(self.skip_outtwo, up_skiptwo, main_up_one)
        main_up_three = self.mainFusionThree(self.skip_outone, up_skipone, main_up_two)
        main_result = self.main_last(main_up_three)

        re_loss = self.L1Loss(re_result, re_down)
        return re_down, mask_image, re_result, main_result, re_loss, x


if __name__ == '__main__':
    model = MyModel(act="relu", base_filter=16)
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    a = torch.rand((1, 1, 512, 512))
    for i in range(1, 2):
        print("epoch : ", i)

        re_down, mask_image, re_result, main_result, re_loss, x = model(a)
        re_loss.backward()
        optimizer.step()
        b = model(a)
        if i % 1 == 0:
            # print(re_loss)
            # 可视化原始图像和处理后的图像
            plt.figure(figsize=(12, 6))
            plt.subplot(131)
            plt.imshow(re_down[0][0], cmap='gray')  # 显示灰度图像
            plt.title("Original Image")
            plt.subplot(132)
            plt.imshow(mask_image[0][0].detach(), cmap='gray')  # 显示处理后的灰度图像
            plt.title("Processed Image")
            plt.subplot(133)
            plt.imshow(re_result[0][0].detach(), cmap='gray')  # 显示处理后的灰度图像
            plt.title("Processed Image")
            plt.show()