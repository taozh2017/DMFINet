import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import random
from torchvision import transforms
import torch.nn.functional as F
import re

def downsample_image(img):
    # Convert NumPy array to PyTorch tensor
    img = torch.tensor(img, dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # Add batch and channel dimensions

    # Downsample the image using PyTorch's interpolate function
    img = F.interpolate(img, scale_factor=0.5, mode='bilinear', align_corners=True)

    # Clip values to be within [0, 255] and convert to uint8  插值之后可能不在0-255了，会导致转为tensor不会归一化
    img = img.clamp(0, 255).byte()

    # Convert back to NumPy array
    img = img.squeeze().numpy()

    return img

def cv_random_flip(mrImg, ctImg, maskImg):
    # 随机水平或垂直翻转
    flip_flag = random.randint(0, 2)
    if flip_flag == 1:
        mrImg = np.flip(mrImg, 0).copy()
        ctImg = np.flip(ctImg, 0).copy()
        maskImg = np.flip(maskImg, 0).copy()
    elif flip_flag == 2:
        mrImg = np.flip(mrImg, 1).copy()
        ctImg = np.flip(ctImg, 1).copy()
        maskImg = np.flip(maskImg, 1).copy()
    return mrImg, ctImg, maskImg

def randomRotation(mrImg, ctImg, maskImg):
    # 随机旋转 0 到 3 个 90 度
    rotate = random.randint(0, 1)
    if rotate == 1:
        rotate_time = random.randint(1, 3)
        mrImg = np.rot90(mrImg, rotate_time).copy()
        ctImg = np.rot90(ctImg, rotate_time).copy()
        maskImg = np.rot90(maskImg, rotate_time).copy()
    return mrImg, ctImg, maskImg

def data_augment(mrImg, ctImg, maskImg, size=256):
    # 随机旋转
    mrImg, ctImg, maskImg = randomRotation(mrImg, ctImg, maskImg)

    # 随机翻转
    mrImg, ctImg, maskImg = cv_random_flip(mrImg, ctImg, maskImg)

    return mrImg, ctImg, maskImg

def load_data(img_name, is_train=True, size=256, argument=True):
    img = cv2.imread(img_name, cv2.IMREAD_GRAYSCALE)
    mrImg, ctImg, maskImg = img[:, :size], img[:, size:2*size], img[:, -size:]

    if argument:
        mrImg, ctImg, maskImg = data_augment(mrImg, ctImg, maskImg)

    # 将掩码图像转换为二值图像
    maskImg[maskImg < 127.5] = 0.
    maskImg[maskImg >= 127.5] = 1.
    maskImg = maskImg.astype(np.uint8)

    return mrImg, ctImg, maskImg

def all_files_under(path, extension='png', append_path=True, sort=True):
    if append_path:
        if extension is None:
            filenames = [os.path.join(path, fname) for fname in os.listdir(path)]
        else:
            filenames = [os.path.join(path, fname) for fname in os.listdir(path) if fname.endswith(extension)]
    else:
        if extension is None:
            filenames = [os.path.basename(fname) for fname in os.listdir(path)]
        else:
            filenames = [os.path.basename(fname) for fname in os.listdir(path) if fname.endswith(extension)]

    if sort:
        filenames = sorted(filenames)

    return filenames

class HanDataset(Dataset):
    def __init__(self, data_dir, is_train=True, argument=True, re_augment=False, re_down=True, data_rate = 1):
        self.data_dir = data_dir
        self.is_train = is_train
        self.argument = argument
        self.re_down = re_down
        self.filenames = all_files_under(self.data_dir, extension='png')
        print(self.filenames)
        self.number_list = [re.search(r'/(\d+)\.', file).group(1) for file in self.filenames]
        self.num_samples = len(self.filenames)
        self.data_rate = data_rate
        
        self.img_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5]),
        ])
        self.mask_transform = transforms.Compose([
            transforms.ToTensor()
        ])

        self.train_len = int(0.8 * self.num_samples)
        self.test_len = self.num_samples - self.train_len

    def __len__(self):
        if self.is_train:
            return self.train_len
        else:
            return int(self.data_rate * self.test_len)

    def __getitem__(self, idx):
        # 从数据集中随机选择两个数据点
        idx_A = idx
        if self.is_train:  # 使其不配对
            idx_B = random.randint(0, len(self.filenames) - 1)  # 随机选择一个不同的索引
        else:
            idx_B = idx_A

        if not self.is_train:
            idx_A, idx_B = idx_A + self.train_len, idx_B + self.train_len
        img_name_A = self.filenames[idx_A]
        img_name_B = self.filenames[idx_B]
        name = self.number_list[idx_A]

        # 加载图像数据
        mrImg, _, mask_A = load_data(img_name_A, is_train=self.is_train, argument=self.argument)  # 只需加载输入图像数据
        _, ctImg, mask_B = load_data(img_name_B, is_train=self.is_train, argument=self.argument)  # 只需加载目标图像数据

        A_down_image = mrImg.copy()
        A_down_image = downsample_image(A_down_image)
        A_re_down_image = mrImg.copy()

        B_down_image = ctImg.copy()
        B_down_image = downsample_image(B_down_image)
        B_re_down_image = ctImg.copy()

        if self.re_down == True:
            A_re_down_image = downsample_image(A_re_down_image)
            B_re_down_image = downsample_image(B_re_down_image)

        mask_A = self.mask_transform(mask_A)
        mask_B = self.mask_transform(mask_B)
        mask_A = 1 - mask_A
        mask_B = 1 - mask_B
        # 将 NumPy 数组转换为 PyTorch 张量
        mrImg = self.img_transform(mrImg)
        ctImg = self.img_transform(ctImg)
        A_down_image = self.img_transform(A_down_image)
        A_re_down_image = self.img_transform(A_re_down_image)
        B_down_image = self.img_transform(B_down_image)
        B_re_down_image = self.img_transform(B_re_down_image)

        return {"A": mrImg, "B": ctImg, "name": name, "A_down_image": A_down_image, "A_re_down_image": A_re_down_image,
                                    "B_down_image": B_down_image, "B_re_down_image": B_re_down_image,"mask_A":mask_A,
                                    "mask_B":mask_B}

def get_loader(batch_size, shuffle, pin_memory=True, data_path = "post", is_train=True, num_workers=1, argument = True, re_augment=False, re_down=True, data_rate=1):
    dataset = HanDataset(data_dir=data_path, is_train = is_train, argument=argument, re_augment=re_augment, re_down=re_down,data_rate=data_rate)
    if not is_train:
        shuffle = False
    data_loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle,
                                  pin_memory=pin_memory, num_workers=num_workers)
    return data_loader


import matplotlib.pyplot as plt
import numpy as np


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

if __name__ == "__main__":
    dataloader = get_loader(batch_size=1, data_path="post", shuffle=True, num_workers=4, is_train=False, argument = False, re_augment=False, re_down=True)

    for batch_idx, data in enumerate(dataloader):
        mrImgs = data["A"]
        ctImgs = data["A_re_down_image"]
        name = data["name"]
        show_two_images(mrImgs, ctImgs, name)




