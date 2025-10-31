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


def cv_resize_and_crop(mrImg, ctImg, maskImg, size):
    # Step 1: 调整图像大小
    if size == 512:
        add = 60
    else:
        add = 30
    mrImg_resized = cv2.resize(mrImg, (size+add, size+add), interpolation=cv2.INTER_LINEAR)
    ctImg_resized = cv2.resize(ctImg, (size+add, size+add), interpolation=cv2.INTER_LINEAR)
    maskImg_resized = cv2.resize(maskImg, (size+add, size+add), interpolation=cv2.INTER_NEAREST)

    # Step 2: 随机裁剪
    # 获取裁剪起始位置
    x_start = random.randint(0, add)
    y_start = random.randint(0, add)

    mrImg_cropped = mrImg_resized[y_start:y_start + size, x_start:x_start + size]
    ctImg_cropped = ctImg_resized[y_start:y_start + size, x_start:x_start + size]
    maskImg_cropped = maskImg_resized[y_start:y_start + size, x_start:x_start + size]

    return mrImg_cropped, ctImg_cropped, maskImg_cropped


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

    mrImg, ctImg, maskImg = cv_resize_and_crop(mrImg, ctImg, maskImg, size=size)

    return mrImg, ctImg, maskImg


def load_data(img_name, size=256, argument=True):
    img = cv2.imread(img_name, cv2.IMREAD_GRAYSCALE)
    mrImg, ctImg, maskImg = img[:, :size], img[:, size:2 * size], img[:, -size:]

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
    def __init__(self, data_dir, mode="train/", argument=True, re_augment=False, re_down=True, data_rate=1):
        self.data_dir = data_dir
        self.mode = mode
        self.argument = argument
        self.re_down = re_down
        self.data_dir = data_dir + self.mode
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

        self.trainA_list = self.filenames
        self.trainB_list = self.filenames

    def shuffle_trainB(self):
        """ 打乱 B 数据集 """
        self.trainB_list = random.sample(self.trainB_list, len(self.trainB_list))
        self.trainA_list = random.sample(self.trainA_list, len(self.trainA_list))
        print("B data shuffled")

    def __len__(self):
        
        return self.num_samples


    def __getitem__(self, idx):

        img_name_A = self.trainA_list[idx]
        img_name_B = self.trainB_list[idx]
        name = self.number_list[idx]

        # 加载图像数据
        mrImg, _, mask_A = load_data(img_name_A, argument=self.argument)  # 只需加载输入图像数据
        _, ctImg, mask_B = load_data(img_name_B, argument=self.argument)  # 只需加载目标图像数据

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
                "B_down_image": B_down_image, "B_re_down_image": B_re_down_image, "mask_A": mask_A,
                "mask_B": mask_B}


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


def get_loader(batch_size, shuffle, pin_memory=True, data_path="raw/post/", mode="train/", num_workers=1,
               argument=True, re_augment=False, re_down=True, data_rate=1):
    dataset = HanDataset(data_dir=data_path, argument=argument, re_augment=re_augment,
                         re_down=re_down, data_rate=data_rate, mode=mode)

    data_loader = CustomDataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle,
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
    dataloader = get_loader(batch_size=1, data_path="raw/post/", mode="train/", shuffle=True, num_workers=4,
                            argument=True,
                            re_augment=False, re_down=True)

    for batch_idx, data in enumerate(dataloader):
        mrImgs = data["A"]
        ctImgs = data["B"]
        name = data["name"]
        # show_two_images(mrImgs, ctImgs, name)