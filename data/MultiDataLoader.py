import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from matplotlib import pyplot as plt
from torchvision import datasets
from torchvision import transforms
from torchvision.utils import save_image
import torch.utils.data as data
import numpy as np
from PIL import ImageEnhance, Image
import random
import os
import pickle
import cv2

import re
from skimage.measure import label
from skimage import (exposure, feature, filters, io, measure,
                     morphology, restoration, segmentation, transform,
                     util)

def get_3d_mask(img, min_, max_=None, th=50, width=2):
    if max_ is None:
        max_ = img.max()
    img = np.clip(img, min_, max_)
    img = np.uint8(255 * img)

    ## Remove artifacts
    mask = np.zeros(img.shape).astype(np.int32)
    mask[img > th] = 1

    ## Remove artifacts and small holes with binary opening
    mask = morphology.binary_opening(mask, )  # 先腐蚀后膨胀

    remove_holes = morphology.remove_small_holes(
        mask,
        area_threshold=width ** 3
    )

    largest_cc = getLargestCC(remove_holes)
    return img, largest_cc.astype(np.int32)


def getLargestCC(segmentation):  # 找到二值图像最大连通组件
    labels = label(segmentation)
    assert (labels.max() != 0)  # assume at least 1 CC
    largestCC = labels == np.argmax(np.bincount(labels.flat)[1:]) + 1
    return largestCC


def extract_numbers_from_paths(paths):
    numbers = []
    for path in paths:
        # 使用正则表达式从文件路径中提取数字
        match = re.search(r'(\d+)\.npy', path)
        if match:
            numbers.append(int(match.group(1)))
    return numbers


# 定义一个自定义的转换
class ToMinusOneToOne(object):
    def __call__(self, tensor):
        return tensor * 2 - 1

def cv_random_flip_one(img):
    # left right flip
    flip_flag = random.randint(0, 2)
    if flip_flag == 1:
        img = np.flip(img, 0).copy()
    if flip_flag == 2:
        img = np.flip(img, 1).copy()  # origin image can't change
    return img

def cv_resize_and_crop(mrImg, size):
    # Step 1: 调整图像大小
    if size == 512:
        add = 60
    else:
        add = 30
    mrImg_resized = cv2.resize(mrImg, (size+add, size+add), interpolation=cv2.INTER_LINEAR)

    x_start = random.randint(0, add)
    y_start = random.randint(0, add)

    mrImg_cropped = mrImg_resized[y_start:y_start + size, x_start:x_start + size]

    return mrImg_cropped


def randomRotation_one(image):
    rotate = random.randint(0, 1)
    if rotate == 1:
        rotate_time = random.randint(1, 3)
        image = np.rot90(image, rotate_time).copy()
    return image

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


def upsample_image(img, size=256):
    # Convert NumPy array to PyTorch tensor
    img = torch.tensor(img, dtype=torch.float).unsqueeze(0).unsqueeze(0)  # Assuming img is a 2D array (H, W)

    # Downsample the image using PyTorch's interpolate function
    img = F.interpolate(img, size=(size, size), mode='bilinear', align_corners=True)

    # Convert back to NumPy array
    img = img.squeeze().numpy()

    return img

def load_npy(image):
    npy = np.load(image, allow_pickle=True)
    # 提取 'archive/data.pkl' 的数据
    pkl_data = npy['archive/data.pkl']
    unpickled_data = pickle.loads(pkl_data)

    return unpickled_data

class MultiDataset(data.Dataset):
    def __init__(self, source_modal, target_modal, img_size,
                 image_root, mode, data_rate=1, sort=False, argument=False, random=False,
                 re_down=True):

        self.source = source_modal
        self.target = target_modal
        self.mode = mode
        self.modal_list = ['ct', 't2']
        self.image_root = image_root + mode
        self.data_rate = data_rate
        self.images = [self.image_root + f for f in os.listdir(self.image_root) if f.endswith('.npy')]
        self.images.sort(key=lambda x: int(x.split(self.image_root)[1].split(".npy")[0]))

        self.names = [self.image_root + f for f in os.listdir(self.image_root) if f.endswith('.npy')]
        self.names.sort(key=lambda x: int(x.split(self.image_root)[1].split(".npy")[0]))
        self.names = extract_numbers_from_paths(self.names)

        self.img_transform = transforms.Compose([
            transforms.ToTensor(),
            ToMinusOneToOne(),
        ])
        self.gt_transform = transforms.Compose([
            transforms.ToTensor(),
            ToMinusOneToOne(),
        ])
        self.mask_transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        self.sort = sort
        self.argument = argument
        self.re_down = re_down
        self.image_size = img_size
        self.train_len = len(self.images)
        self.trainA_list = self.images
        self.trainB_list = self.images
        self.mark = 0

    def shuffle_trainB(self):
        """ 打乱 B 数据集 """
        self.trainB_list = random.sample(self.trainB_list, len(self.trainB_list))
        self.trainA_list = random.sample(self.trainA_list, len(self.trainA_list))
        print("B data shuffled")

    def __getitem__(self, index):
        npy_one = load_npy(self.trainA_list[index])
        npy_two = load_npy(self.trainB_list[index])
        img = npy_one[self.modal_list.index(self.source), :, :]
        gt = npy_two[self.modal_list.index(self.target), :, :]  # 0-1

        img = upsample_image(img, self.image_size)
        gt = upsample_image(gt, self.image_size)
        name = self.names[index]

        if self.argument == True:
            img, gt = cv_random_flip_one(img), cv_random_flip_one(gt)
            img, gt = randomRotation_one(img), randomRotation_one(gt)
            # img, gt = cv_resize_and_crop(img, size=self.image_size), cv_resize_and_crop(gt, size=self.image_size)

        A_down_image = img.copy()
        A_down_image = downsample_image(A_down_image)
        A_re_down_image = img.copy()

        B_down_image = gt.copy()
        B_down_image = downsample_image(B_down_image)
        B_re_down_image = gt.copy()

        if self.re_down == True:
            # print("downing")
            A_re_down_image = downsample_image(A_re_down_image)
            B_re_down_image = downsample_image(B_re_down_image)

        # 添加MaSKgAN的方法
        img_A, mask_A = get_3d_mask(img, min_=0, max_=1, th=10, width=10)
        gt_B, mask_B = get_3d_mask(gt, min_=0, max_=1, th=10, width=10)

        img = self.img_transform(img)
        gt = self.img_transform(gt)

        mask_A = self.mask_transform(mask_A)
        mask_B = self.mask_transform(mask_B)

        mask_A = 1 - mask_A
        mask_B = 1 - mask_B

        A_down_image = self.img_transform(A_down_image)
        A_re_down_image = self.img_transform(A_re_down_image)
        B_down_image = self.img_transform(B_down_image)
        B_re_down_image = self.img_transform(B_re_down_image)

        return {"A": img, "B": gt, "A_down_image": A_down_image, "A_re_down_image": A_re_down_image,
                "B_down_image": B_down_image, "B_re_down_image": B_re_down_image, "name": name, "mask_A": mask_A,
                "mask_B": mask_B}

    def __len__(self):
        return int(len(self.images) * self.data_rate)

from torch.utils.data import DataLoader

class CustomDataLoader(DataLoader):
    def __init__(self, dataset, batch_size, shuffle, pin_memory=True, num_workers=1):
        super().__init__(dataset=dataset, batch_size=batch_size, shuffle=shuffle,
                         pin_memory=pin_memory, num_workers=num_workers)

    def __iter__(self):
        # 在每个 epoch 开始时打乱 B 数据集
        if self.dataset.mode == "train/":
            self.dataset.shuffle_trainB()
        return super().__iter__()

def get_loader(batchsize, shuffle, pin_memory=True, source_modal='t2', target_modal='ct',
               img_size=512, img_root='data/', mode="train/", data_rate=1, num_workers=8, sort=False, argument=False,
               random=False, re_down=True):
    dataset = MultiDataset(source_modal=source_modal, target_modal=target_modal,
                            img_size=img_size, image_root=img_root, mode=mode, data_rate=data_rate, sort=sort,
                            argument=argument, random=random, re_down=re_down)

    data_loader = CustomDataLoader(dataset=dataset, batch_size=batchsize, shuffle=shuffle,
                                   pin_memory=pin_memory, num_workers=num_workers)
    return data_loader


def show_two_images(tensor1, tensor2, name):
    # Convert from tensor to numpy
    img1 = tensor1.squeeze().numpy()  # Remove channel dimension and convert to numpy array
    img2 = tensor2.squeeze().numpy()  # Remove channel dimension and convert to numpy array

    # Create subplots
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))

    # Display images
    axs[0].imshow(img1, cmap='gray')  # Display first image
    axs[0].set_title(name)
    axs[0].axis('off')  # Hide axes

    axs[1].imshow(img2, cmap='gray')  # Display second image
    axs[1].set_title("Image 2")
    axs[1].axis('off')  # Hide axes

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    dataloader = get_loader(batchsize=1, shuffle=True, pin_memory=True, source_modal='t2',
                             target_modal='ct', img_size=512, num_workers=8,
                             img_root='../multiData/', mode="train/", data_rate=1, argument=True, random=False, re_down=True)

    length = len(dataloader)
    print("data_loader的长度为:", length)
    a=0
    for batch_idx, data in enumerate(dataloader):
        mrImgs = data["A"]
        ctImgs = data["B"]
        name = data["name"]
        show_two_images(mrImgs, ctImgs, name)



