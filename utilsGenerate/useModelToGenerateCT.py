import glob
import numpy as np
import pandas as pd
import torch
import re
import os
import SimpleITK as sitk
import nibabel as nib

from models.networks import define_G
import torch.nn.functional as F
device = torch.device('cuda:{}'.format(1))
def downsample_image(img):

    img = F.interpolate(img, scale_factor=0.5, mode='bilinear', align_corners=True)

    return img


def upsample_image(img, size=256):

    # Downsample the image using PyTorch's interpolate function
    img = F.interpolate(img, size=(size, size), mode='bilinear', align_corners=True)

    return img

def normalize(image, percentile_lower=0.1, percentile_upper=99.98):
    cut_off_lower = np.percentile(image.ravel(), percentile_lower)
    cut_off_upper = np.percentile(image.ravel(), percentile_upper)
    res = np.copy(image)
    res[res < cut_off_lower] = cut_off_lower
    res[res > cut_off_upper] = cut_off_upper
    a = np.min(res)
    b = np.max(res)
    res = (res - a) / (b - a)  # 0-1  version_one
    return res

def process(path, save_path, netG):
    print(path)
    match = re.search(r'MR_copy/(.*?)\/Processed Images', path)
    name = "name"
    if match:
        name = match.group(1)  # 提取匹配的部分
    else:
        print("No match found.")
    print(name)
    image_size = 512
    # 获取 MR 文件名列表并排序
    mr_file_names = os.listdir(path)
    sorted_mr_files = sorted(mr_file_names, key=lambda x: int(x.split("IM")[1]))
    mr_full_paths = [os.path.join(path, file) for file in sorted_mr_files]
    # 读取 MR DICOM 序列
    mr_reader = sitk.ImageSeriesReader()
    mr_reader.SetFileNames(mr_full_paths)
    mr_image3D = mr_reader.Execute()
    mr_array = sitk.GetArrayFromImage(mr_image3D)

    print("MR array shape:", mr_array.shape)
    ct_tensor = np.zeros((mr_array.shape[0], 508, 508))

    for i in range(mr_array.shape[0]):
        print("process: ", i, "  total:  ", mr_array.shape[0])
        one_image = mr_array[i, :, :]
        print(one_image.shape)
        one_image = normalize(one_image, percentile_lower=0.1, percentile_upper=99.98)
        tensor = torch.from_numpy(one_image)
        tensor = tensor.unsqueeze(0).unsqueeze(0)  # 添加两个维度
        tensor = upsample_image(tensor, image_size)
        down_image = downsample_image(tensor).float().to(device)
        re_down_image = downsample_image(tensor).float().to(device)
        tensor = tensor.float()
        tensor = tensor * 2 - 1
        tensor = tensor.to(device)
        _, _, result, _ = netG(tensor, down_image, re_down_image)
            # 删除不再使用的变量并释放缓存
        result = upsample_image(result, 508)
        result = result.squeeze().cpu()
        result = (result + 1) / 2
        ct_tensor[i, :, :] = result.detach().numpy() * 255

    affine = np.eye(4)
    ct_tensor_numpy = np.rot90(ct_tensor, k=1, axes=(1, 2))  # 90度旋转
    ct_tensor_numpy = np.flip(ct_tensor_numpy, axis=2)  # 沿指定轴翻转

    ct_tensor_numpy = np.transpose(ct_tensor_numpy, (1, 2, 0))
    ct_tensor_numpy = np.flip(ct_tensor_numpy, [2])

    ct_img = nib.Nifti1Image(ct_tensor_numpy, affine=affine)
    # 保存为Nii文件
    ctSavePath = str(save_path + name + ".nii.gz")
    nib.save(ct_img, ctSavePath)

# helper loading function that can be used by subclasses
def load_network(network, network_label, epoch_label):
    # save_filename = '%s_net_%s.pth' % (epoch_label, network_label)
    save_filename = "en2ctTest/5_net_G_A.pth"
    save_path = os.path.join("checkpoints_use/", save_filename)
    print(save_path)
    #save_path = save_filename
    network.load_state_dict(torch.load(save_path), strict=False)

    return network

if __name__ == "__main__":

    mrList = sorted(glob.glob("../../data/MR_copy/*/Processed Images/"))
    print(mrList)
    save_path = "converted CT/"
    os.makedirs(save_path, exist_ok=True)
    netG = define_G(8, 0, 64, "mymodel", "instance",
                                    [0, ], "True", "aaa", 3, 3)
    
    netG = load_network(netG, 100, "en2ct1/")
    netG.to(device)
    for i in range(len(mrList)):
        print("process: ", mrList[i])

        process(mrList[i], save_path, netG)