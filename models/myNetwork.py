import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from matplotlib import pyplot as plt
from data.EN2CTDataloader import get_loader
from .Attention.myLargeKernel import LargeConv, LargeConvSpatial, SimpleEnhance, LargeConv1
from .Attention.CBAM import CBAM
from .Attention.LSKBlock import LSKblock
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
    def __init__(self, in_chan, out_chan, act, norm_layer, use_bias, padding_mode="zeros", kernel_size=3, padding=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = LUConv(in_chan, out_chan, act, norm_layer, use_bias, padding_mode, kernel_size, padding)
        self.conv2 = nn.Conv2d(out_chan, out_chan, kernel_size=kernel_size, padding=padding, bias=use_bias,
                               padding_mode=padding_mode)
        self.bn2 = norm_layer(out_chan)

        if in_chan != out_chan:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_chan, out_chan, kernel_size=kernel_size, stride=1, bias=use_bias, padding=padding,
                          padding_mode=padding_mode),
                norm_layer(out_chan)
            )
        else:
            self.shortcut = nn.Identity()

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
        # return self.activation(out)
        return out


def _make_nConv(in_channel, out_cha, act, res_net=False, norm_layer=nn.BatchNorm2d, use_bias=False,
                padding_mode="reflect", kernel_size=3, padding=1):
    if res_net:
        return ResidualBlock(in_channel, out_cha, act, norm_layer, use_bias, padding_mode, kernel_size, padding)
    else:
        layer1 = LUConv(in_channel, out_cha, act, norm_layer, use_bias, padding_mode, kernel_size, padding)
        layer2 = LUConv(out_cha, out_cha, act, norm_layer, use_bias, padding_mode, kernel_size, padding)
        return nn.Sequential(layer1, layer2)


class DownTransition(nn.Module):
    def __init__(self, in_channel, out_cha, act, res_net=False, norm_layer=nn.BatchNorm2d, use_bias=False,
                 padding_mode="reflect", kernel_size=3, padding=1, down=True):
        super(DownTransition, self).__init__()

        self.ops = LUConv(in_channel, out_cha, act, norm_layer, use_bias, padding_mode, kernel_size, padding)

        # self.maxpool = nn.MaxPool2d(2)
        self.down = DownsampeModel(in_channel, out_cha, mode="conv", norm_layer=norm_layer, use_bias=use_bias)

    def forward(self, x):
        # out_before_pool = self.ops(x)
        out_before_pool = x
        out = self.down(out_before_pool)

        return out, out_before_pool


class LUConv(nn.Module):
    def __init__(self, in_chan, out_chan, act, norm_layer, use_bias, padding_mode="reflect", kernel_size=3, padding=1):
        super(LUConv, self).__init__()
        self.conv1 = nn.Conv2d(in_chan, out_chan, kernel_size=kernel_size, padding=padding, bias=use_bias,
                               padding_mode=padding_mode)
        self.bn1 = norm_layer(out_chan)

        if act == 'relu':
            self.activation = nn.ReLU(True)
        elif act == 'prelu':
            self.activation = nn.PReLU(num_parameters=out_chan)
        elif act == 'elu':
            self.activation = nn.ELU(inplace=True)
        else:
            raise ValueError(f"Unsupported activation function: {act}")


    def forward(self, x):
        out = self.activation(self.bn1(self.conv1(x)))
        return out


class DownsampeModel(nn.Module):
    def __init__(self, in_cha, out_cha, mode="conv", norm_layer=nn.BatchNorm2d, use_bias=False):
        super(DownsampeModel, self).__init__()
        # 定义转置卷积层等操作
        self.mode = mode
        self.down = None
        if self.mode == "conv":
            self.down = nn.Sequential(nn.Conv2d(in_cha, out_cha, kernel_size=3, stride=2, padding=1, bias=use_bias),
                                      norm_layer(out_cha),
                                      nn.ReLU(True))

    def forward(self, x):
        return self.down(x)


class UpLayer(nn.Module):
    def __init__(self, in_cha, out_cha, res_net=False, norm_layer=nn.BatchNorm2d, use_bias=False,
                 padding_mode="reflect"):
        super(UpLayer, self).__init__()

        self.model = nn.Sequential(nn.ConvTranspose2d(in_cha, out_cha,
                                                      kernel_size=3, stride=2,
                                                      padding=1, output_padding=1,
                                                      bias=use_bias),
                                   norm_layer(out_cha),
                                   nn.ReLU(True))
        # self.layer = _make_nConv(out_cha, out_cha, "relu", res_net, norm_layer, use_bias=use_bias, padding_mode=padding_mode)
        # self.layer = LUConv(in_cha, out_cha, "relu", norm_layer, use_bias, padding_mode, kernel_size=3, padding=1)

    def forward(self, x):
        # 使用反卷积层进行上采样操作
        x = self.model(x)
        # x = self.layer(x)
        return x


class mainFusion(nn.Module):
    def __init__(self, in_cha, out_cha, res_net=False, norm_layer=nn.BatchNorm2d, use_bias=False,
                 padding_mode="reflect"):
        super(mainFusion, self).__init__()
        # 在这里可以定义需要初始化的层，如反卷积层等
        self.conv1 = nn.Sequential(nn.Conv2d(in_cha, out_cha, kernel_size=1, stride=1, padding=0, bias=use_bias),
                                   norm_layer(out_cha),
                                   nn.ReLU(True))   

        self.up1 = nn.Sequential(nn.ConvTranspose2d(in_cha, out_cha,
                                                    kernel_size=3, stride=2,
                                                    padding=1, output_padding=1,
                                                    bias=use_bias),
                                 norm_layer(out_cha),
                                 nn.ReLU(True))
        self.up2 = nn.Sequential(nn.ConvTranspose2d(out_cha, out_cha,
                                                    kernel_size=3, stride=2,
                                                    padding=1, output_padding=1,
                                                    bias=use_bias),
                                 norm_layer(out_cha),
                                 nn.ReLU(True))
        self.att = AttentionGate(out_cha, out_cha, int(out_cha / 2))
        # self.att = SimpleEnhance(out_cha)
        self.conv2 = LUConv(out_cha, out_cha, "relu", norm_layer, use_bias, padding_mode, kernel_size=3, padding=1)


    def forward(self, main_skip, up_skip, main_decoder):  # 分别是主要分支对应得skipconnectin， 上面分支对应的skip, 主分支下层特征
        x = self.up1(main_decoder)  # 128 256 256
        a = main_skip
        b = main_decoder
        c = up_skip
        d = x
        y = self.up2(up_skip)  #128 256 256
        e = y
        y = torch.cat([y, main_skip], dim=1)  # 256 256 256
        y = self.conv1(y)  # 128 256 256
        f = y

        # y_skip = self.att(x, y) # 128 256 256
        y_skip = self.att(x, y) # 128 256 256
        xy = y_skip + x  # 128 256 256
        g = xy
        xy = self.conv2(xy)
        h = xy

        vis = [a, b, c, d, e, f, g, h]
        return xy, vis


def maskRand(img, patch_size, mask_ratio):
    img = patchify(img, patch_size)  # 1 1024 256
    # masking: length -> length * mask_ratio
    img = (img + 1) / 2  # -1 1  -> 0 1
    x, mask, ids_restore, mask_image, mask_expand = random_masking(img, mask_ratio)
    mask_image = (mask_image * 2) - 1  # 0 1 -> -1 1
    mask_image = unpatchify(mask_image, patch_size=patch_size)

    return x, mask, ids_restore, mask_image, mask_expand


class MyModel(nn.Module):
    def __init__(self, act='relu', base_filter=64, res_bottleneck_layers=6, large_num=1, re_outcha=1,
                 patch_size=8, mask_ratio=0.2, res_net=False, norm_layer=nn.BatchNorm2d, att_dim=256,
                 padding_mode="reflect", n_att=5):
        super(MyModel, self).__init__()
        self.patch_size = patch_size
        self.mask_ratio = mask_ratio
        self.base_filter = base_filter
        self.att_dim = att_dim
        self.padding_mode = padding_mode
        self.n_att = n_att
        self.output_nc = re_outcha
        print("layers  res:", res_bottleneck_layers)
        print("large num: ", large_num)
        print("res_net value :  ", res_net)
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        print("using biass : ", use_bias)

        # 权重共享
        self.main_first = LUConv(1, self.base_filter, act, norm_layer, use_bias, padding_mode, kernel_size=7, padding=3)

        self.down_one = DownTransition(self.base_filter, self.base_filter * 2, act, res_net=res_net, norm_layer=norm_layer,
                                       use_bias=use_bias, padding_mode=self.padding_mode)
        self.down_two = DownTransition(self.base_filter * 2, self.base_filter * 4, act, res_net=res_net,
                                       norm_layer=norm_layer, use_bias=use_bias, padding_mode=self.padding_mode)
        # self.down_three = DownTransition(self.base_filter * 2, self.base_filter * 4, act, res_net=res_net,
        #                                  norm_layer=norm_layer, use_bias=use_bias, padding_mode=self.padding_mode, down=False)
        self.bottle_neck_channel = self.base_filter * 4

        # 重建模块
        # 第一次的底层，两者没有交互。
        self.re_bottlneck = []
        for _ in range(res_bottleneck_layers):
            self.re_bottlneck.append(
                _make_nConv(self.bottle_neck_channel, self.bottle_neck_channel, act, res_net=res_net,
                            norm_layer=norm_layer, use_bias=use_bias, padding_mode=self.padding_mode))
        for _ in range(large_num):
            self.re_bottlneck.append(LargeConv(self.bottle_neck_channel))
            # self.re_bottlneck.append(LSKblock(self.bottle_neck_channel))
            # self.re_bottlneck.append(LargeConvSpatial(self.bottle_neck_channel))
        self.re_bottlneck_model = nn.Sequential(*self.re_bottlneck)

        # 上面特征提取模块 参数不共享
        self.up_first = LUConv(1, self.base_filter, act, norm_layer, use_bias, padding_mode, kernel_size=7, padding=3)
        self.up_down_one = DownTransition(self.base_filter, self.base_filter * 2, act, res_net=res_net, norm_layer=norm_layer,
                                          use_bias=use_bias, padding_mode=self.padding_mode)
        self.up_down_two = DownTransition(self.base_filter * 2, self.base_filter * 4, act, res_net=res_net,
                                          norm_layer=norm_layer, use_bias=use_bias, padding_mode=self.padding_mode)

        # # 主分支模块
        # 第一次，没有交互
        self.main_bottlneck = []
        for _ in range(res_bottleneck_layers):
            self.main_bottlneck.append(
                _make_nConv(self.bottle_neck_channel, self.bottle_neck_channel, act, res_net=res_net,
                            norm_layer=norm_layer, use_bias=use_bias, padding_mode=self.padding_mode))
        for _ in range(large_num):
            self.main_bottlneck.append(LargeConv(self.bottle_neck_channel))
            # self.main_bottlneck.append(LSKblock(self.bottle_neck_channel))
            # self.main_bottlneck.append(LargeConvSpatial(self.bottle_neck_channel))
        self.main_bottlneck_model = nn.Sequential(*self.main_bottlneck)

        self.mainFusionTwo = mainFusion(self.base_filter * 4, self.base_filter * 2, res_net=res_net, norm_layer=norm_layer,
                                        use_bias=use_bias, padding_mode=self.padding_mode)
        self.mainFusionThree = mainFusion(self.base_filter * 2, self.base_filter * 1, res_net=res_net,
                                          norm_layer=norm_layer, use_bias=use_bias, padding_mode=self.padding_mode)
        self.main_last = nn.Sequential(nn.ReflectionPad2d(3),
                                       nn.Conv2d(self.base_filter * 1, re_outcha, kernel_size=7, padding=0),
                                       nn.Tanh())

        # upsample reconstruction
        # self.re_upone = UpLayer(self.base_filter * 4, self.base_filter * 2, res_net=res_net, norm_layer=norm_layer,
        #                         use_bias=use_bias, padding_mode=self.padding_mode)
        self.re_uptwo = UpLayer(self.base_filter * 4, self.base_filter * 2, res_net=res_net, norm_layer=norm_layer,
                                use_bias=use_bias, padding_mode=self.padding_mode)
        self.re_upthree = UpLayer(self.base_filter * 2, self.base_filter * 1, res_net=res_net,
                                  norm_layer=norm_layer, use_bias=use_bias, padding_mode=self.padding_mode)
        
        # for delete MSFF
        self.main_uptwo = UpLayer(self.base_filter * 4, self.base_filter * 2, res_net=res_net, norm_layer=norm_layer,
                                use_bias=use_bias, padding_mode=self.padding_mode)
        self.main_upthree = UpLayer(self.base_filter * 2, self.base_filter * 1, res_net=res_net,
                                  norm_layer=norm_layer, use_bias=use_bias, padding_mode=self.padding_mode)

        self.re_last = nn.Sequential(nn.ReflectionPad2d(3),
                                     nn.Conv2d(self.base_filter * 1, re_outcha, kernel_size=7, padding=0),
                                     nn.Tanh())
        self.L1Loss = nn.L1Loss()

    def forward(self, x, down_image, re_down_image):

        # process the main branch
        self.first = self.main_first(x)
        self.out_one, self.skip_outone = self.down_one(self.first)
        self.out_two, self.skip_outtwo = self.down_two(self.out_one)

        # process the reconstruct branch
        x_masked, mask, ids_restore, mask_image, mask_expand = maskRand(img=re_down_image, patch_size=self.patch_size,
                                                                        mask_ratio=self.mask_ratio)
        # print(self.patch_size)
        # print(self.mask_ratio)
        re_first = self.main_first(mask_image)
        re_one, re_skipone = self.down_one(re_first)
        re_two, re_skiptwo = self.down_two(re_one)

        # process bottleneck fusion
        # TWO
        self.main_bottle = self.main_bottlneck_model(self.out_two)
        self.re_bottle = self.re_bottlneck_model(re_two)

        # process the reconstruction
        up = self.re_bottle
        up = self.re_uptwo(up)
        up = self.re_upthree(up)
        re_result = self.re_last(up)

        # process the up branch
        up_first = self.up_first(down_image)
        up_one, up_skipone = self.up_down_one(up_first)  # 1 32 128 128    1 32 256 256
        up_two, up_skiptwo = self.up_down_two(up_one)

        # process the main up branch
        main_up = self.main_bottle

        main_up_two, vis1 = self.mainFusionTwo(self.skip_outtwo, up_skiptwo, main_up)
        main_up_three, vis1 = self.mainFusionThree(self.skip_outone, up_skipone, main_up_two)
        # 去掉MSFF
        # main_up_two = self.main_uptwo(main_up)
        # main_up_three = self.main_upthree(main_up_two)
        vis1 = main_up_three
        main_result = self.main_last(main_up_three)
        # vis1 = main_result
        return mask_image, re_result, main_result, vis1
        #return mask_image, main_result, main_result


if __name__ == '__main__':
    model = MyModel(act="relu", base_filter=64, mask_ratio=0.2, patch_size=8, res_net=True,
                    norm_layer=nn.InstanceNorm2d,
                    att_dim=256, padding_mode="reflect")

    optimizer = optim.SGD(model.parameters(), lr=0.01)
    a = torch.rand((1, 1, 512, 512))
    a[:, :, :200, :] = 1.0
    a[:, :, 200:400, :] = 0.5
    a[:, :, 400:, :] = 0
    data_loader = get_loader(batchsize=1, shuffle=False, pin_memory=True, source_modal='en', target_modal='ct',
                             img_size=512, img_root="../data/", model="train/", data_rate=1, num_workers=24, sort=False,
                             argument=True,
                             random=False, re_augment=True, re_down=True)


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
        mask_image, re_result, main_result, _ = model(img, down_image, re_down_image)
        end_time = time.time()
        print("using time: ", end_time - start_time)
        if i % 1 == 0:
            # print(re_loss)
            # 可视化原始图像和处理后的图像
            plt.figure(figsize=(12, 6))
            plt.subplot(141)
            plt.imshow(img[0][0], cmap='gray')  # 显示灰度图像
            plt.title("Original Image")
            plt.subplot(142)
            plt.imshow(mask_image[0][0].detach(), cmap='gray')  # 显示处理后的灰度图像
            plt.title("Processed Image")
            plt.subplot(143)
            plt.imshow(re_result[0][0].detach(), cmap='gray')  # 显示处理后的灰度图像
            plt.title("Processed Image")
            plt.subplot(144)
            plt.imshow(main_result[0][0].detach(), cmap='gray')  # 显示处理后的灰度图像
            plt.title("main Image")
            plt.show()