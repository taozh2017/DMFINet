import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets
from torchvision import transforms
from torchvision.utils import save_image
import torch.utils.data as data
import numpy as np
from PIL import ImageEnhance, Image
import random
import os

from util.networkUtil import augment


# 定义一个自定义的转换
class ToMinusOneToOne(object):
    def __call__(self, tensor):
        return tensor * 2 - 1
    
def cv_random_flip(img, label):
    # left right flip
    flip_flag = random.randint(0, 2)
    if flip_flag == 1:
        img = np.flip(img, 0).copy()
        label = np.flip(label, 0).copy()
    if flip_flag == 2:
        img = np.flip(img, 1).copy()  # origin image can't change
        label = np.flip(label, 1).copy()
    return img, label


def randomRotation(image, label):
    rotate = random.randint(0, 1)
    if rotate == 1:
        rotate_time = random.randint(1, 3)
        image = np.rot90(image, rotate_time).copy()
        label = np.rot90(label, rotate_time).copy()
    return image, label

def downsample_image(img):
    # Convert NumPy array to PyTorch tensor
    img = torch.tensor(img, dtype=torch.float).unsqueeze(0).unsqueeze(0)  # Assuming img is a 2D array (H, W)

    # Downsample the image using PyTorch's interpolate function
    img = F.interpolate(img, scale_factor=0.5, mode='bilinear', align_corners=True)

    # Convert back to NumPy array
    img = img.squeeze().numpy()

    return img

class MR2CT_Dataset(data.Dataset):
    def __init__(self, source_modal, target_modal, img_size,
                 image_root, model, data_rate, sort=False, argument=False, random=False, re_augment=False, re_down=False):

        self.source = source_modal
        self.target = target_modal
        self.modal_list = ['ct', 'en', 'fat']
        self.image_root = image_root + model
        self.data_rate = data_rate
        self.images = [self.image_root + f for f in os.listdir(self.image_root) if f.endswith('.npy')]
        self.images.sort(key=lambda x: int(x.split(self.image_root)[1].split(".npy")[0]))
        self.img_transform = transforms.Compose([
            transforms.ToTensor(),
            ToMinusOneToOne(),
            #transforms.Resize(img_size, Image.NEAREST)
        ])
        self.gt_transform = transforms.Compose([
            transforms.ToTensor(),
            ToMinusOneToOne(),
        ])
        self.sort = sort
        self.argument = argument
        self.re_augment = re_augment
        self.re_down = re_down

    def __getitem__(self, index):
        npy = np.load(self.images[index])
        img = npy[self.modal_list.index(self.source), :, :]
        gt = npy[self.modal_list.index(self.target), :, :]

        if self.argument == True:
            img, gt = cv_random_flip(img, gt)
            img, gt = randomRotation(img, gt)

        A_down_image = img.copy()
        A_down_image = downsample_image(A_down_image)
        A_re_down_image = img.copy()

        B_down_image = gt.copy()
        B_down_image = downsample_image(B_down_image)
        B_re_down_image = gt.copy()

        if self.re_down == True:
            A_re_down_image = downsample_image(A_re_down_image)
            B_re_down_image = downsample_image(B_re_down_image)
        if self.re_augment == True:
            A_re_down_image = augment(A_re_down_image)
            B_re_down_image = augment(B_re_down_image)
        
        img = self.img_transform(img)
        gt = self.img_transform(gt)
        A_down_image = self.img_transform(A_down_image)
        A_re_down_image = self.img_transform(A_re_down_image)
        B_down_image = self.img_transform(B_down_image)
        B_re_down_image = self.img_transform(B_re_down_image)

        A_down_image = A_down_image.float()
        B_down_image = B_down_image.float()
        A_re_down_image = A_re_down_image.float()
        B_re_down_image = B_re_down_image.float()

        
        return {"A": img, "B": gt, "A_down_image": A_down_image, "A_re_down_image": A_re_down_image,
                                    "B_down_image": B_down_image, "B_re_down_image": B_re_down_image}

    def __len__(self):
        return int(len(self.images) * self.data_rate)


def get_loader(batchsize, shuffle, pin_memory=True, source_modal='fat', target_modal='ct',
               img_size=256, img_root='data/', model="train/", data_rate=1, num_workers=8, sort=False, argument=False,
               random=False, re_augment=False, re_down=False):
    dataset = MR2CT_Dataset(source_modal=source_modal, target_modal=target_modal,
                            img_size=img_size, image_root=img_root, model=model, data_rate=data_rate, sort=sort,
                            argument=argument, random=random, re_augment=re_augment, re_down=re_down)
    data_loader = data.DataLoader(dataset=dataset, batch_size=batchsize, shuffle=shuffle,
                                  pin_memory=pin_memory, num_workers=num_workers)
    return data_loader


if __name__ == '__main__':
    data_loader = get_loader(batchsize=1, shuffle=False, pin_memory=True, source_modal='fat',
                             target_modal='ct', img_size=256, num_workers=8,
                             img_root='../dataProcessed/', model="train/", data_rate=1, argument=True, random=False)
    length = len(data_loader)
    print("data_loader的长度为:", length)
    # 将 data_loader 转换为迭代器
    data_iter = iter(data_loader)

    # 获取第一批数据
    batch = next(data_iter)
    try:
        # 打印第一批数据的大小
        print("第一批数据的大小:", batch["t1"].shape)  # 输入图像的张量
        print("第一批数据的大小:", batch["t2"].shape)  # 目标图像的张量

    except  KeyError:
        print(KeyError)

    print(batch)



