from functools import partial
import random
import torch
import torch.nn as nn
import numpy as np
from PIL import ImageEnhance, Image
from timm.models.vision_transformer import PatchEmbed, Block
import torch.nn.functional as F
import torchvision.transforms as transforms

def patchify(imgs, patch_size):
    """
    imgs: (N, 1, H, W)
    x: (N, L, patch_size**2 *1)
    """
    p = patch_size
    assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

    h = w = imgs.shape[2] // p
    x = imgs.reshape(shape=(imgs.shape[0], 1, h, p, w, p))
    x = torch.einsum('nchpwq->nhwpqc', x)
    x = x.reshape(shape=(imgs.shape[0], h * w, p ** 2 * 1))
    return x


def unpatchify(x, patch_size):
    """
    x: (N, L, patch_size**2 *1)
    imgs: (N, 1, H, W)
    """
    p = patch_size
    h = w = int(x.shape[1] ** .5)
    assert h * w == x.shape[1]

    x = x.reshape(shape=(x.shape[0], h, w, p, p, 1))
    x = torch.einsum('nhwpqc->nchpwq', x)
    imgs = x.reshape(shape=(x.shape[0], 1, h * p, h * p))
    return imgs


def random_masking(x, mask_ratio):
    """
    Perform per-sample random masking by per-sample shuffling.
    Per-sample shuffling is done by argsort random noise.
    x: [N, L, D], sequence
    """
    N, L, D = x.shape  # batch, length, dim
    len_keep = int(L * (1 - mask_ratio))

    noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]

    # sort noise for each sample
    ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
    ids_restore = torch.argsort(ids_shuffle, dim=1)

    # keep the first subset
    ids_keep = ids_shuffle[:, :len_keep]
    x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

    # generate the binary mask: 0 is keep, 1 is remove
    mask = torch.ones([N, L], device=x.device)
    mask[:, :len_keep] = 0  # save part
    # unshuffle to get the binary mask
    mask = torch.gather(mask, dim=1, index=ids_restore)

    # Apply mask to the input to get the masked image
    mask_expanded = 1 - mask.unsqueeze(-1).repeat(1, 1, D)   # 1 save part  1 1024 256

    mask_image = x * mask_expanded

    return x_masked, mask, ids_restore, mask_image, mask_expanded


def cv_random_flip(img):
    # left right flip
    flip_flag = random.randint(0, 2) # get 0 1 2 randomly
    if flip_flag == 1:
        img = np.flip(img, 0).copy()
    if flip_flag == 2:
        img = np.flip(img, 1).copy()  # origin image can't change

    return img


def randomRotation(image):
    rotate = random.randint(0, 1)  # get 0 1 randomly
    if rotate == 1:
        rotate_time = random.randint(1, 3)
        image = np.rot90(image, rotate_time).copy()

    return image

def colorEnhance(image):
    bright_intensity = random.randint(7, 13) / 10.0
    image = ImageEnhance.Brightness(image).enhance(bright_intensity)
    contrast_intensity = random.randint(4, 11) / 10.0
    image = ImageEnhance.Contrast(image).enhance(contrast_intensity)
    color_intensity = random.randint(7, 13) / 10.0
    image = ImageEnhance.Color(image).enhance(color_intensity)
    sharp_intensity = random.randint(7, 13) / 10.0
    image = ImageEnhance.Sharpness(image).enhance(sharp_intensity)
    return image

def augment(img):
    img = cv_random_flip(img)
    img = randomRotation(img)
    # img = img * 255
    # img = Image.fromarray(img.astype(np.uint8))
    # img = colorEnhance(img)
    # img = img.convert('L')
    #img = np.array(img)  # 将PIL图像转换为NumPy数组

    return img


def augmentTensor(img):
    # Define transformations
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),  # Random horizontal flip with probability 0.5
        transforms.RandomRotation(degrees=(-10, 10))  # Random rotation between -10 to 10 degrees
    ])

    # Apply transformations to each image in the batch
    img_list = []
    for i in range(img.shape[0]):
        img_transformed = transform(img[i])
        img_list.append(img_transformed)

    # Stack into a tensor
    img = torch.stack(img_list)

    return img

def downsample_image(img):
    # Downsample the image using PyTorch's interpolate function
    img = F.interpolate(img, scale_factor=0.5, mode='bilinear', align_corners=True)

    return img

def getDownAndRe(img, re_down=False, re_augment=False):
    # Clone img to ensure gradients can flow back to it
    img_clone = img.clone()
    down_img = downsample_image(img_clone)

    # Clone img again for re_down_img to ensure no interference
    re_down_img = img_clone.clone()
    if re_down == True:
        re_down_img = downsample_image(re_down_img)
        
    if re_augment == True:
        re_down_img = augmentTensor(re_down_img)

    return down_img, re_down_img