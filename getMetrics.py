import os
import cv2  # 用于读取图像
import numpy as np
import numpy as np
from math import log10, sqrt
from skimage.metrics import structural_similarity as ssim

def psnr(res, gt):
    mse = np.mean((res - gt) ** 2)
    if mse == 0:
        return 100  # Perfect match
    max_pixel = 1
    return 20 * log10(max_pixel / sqrt(mse))

def nmse(res, gt):
    norm_gt = np.linalg.norm(gt, ord='fro') ** 2
    if norm_gt == 0:
        return 0  # Avoid division by zero
    return np.linalg.norm(res - gt, ord='fro') ** 2 / norm_gt

def calculateMetrics(pred, gt):
    pred, gt = (pred + 1) / 2, (gt + 1) / 2  # Normalize to 0-1 range
    a = ssim(pred, gt, data_range=1)
    b = psnr(pred, gt)
    c = nmse(pred, gt)

    return [a, b, c]

def load_image(image_path):
    return cv2.imread(image_path, cv2.IMREAD_GRAYSCALE) / 255.0  # 读取并归一化图像

def process_folders(folder_a, folder_b, flag):
    metrics = {'ssim': [], 'psnr': [], 'nmse': []}

    files_a = sorted(os.listdir(folder_a))
    files_b = sorted(os.listdir(folder_b))

    for file_a, file_b in zip(files_a, files_b):
        path_a = os.path.join(folder_a, file_a)
        path_b = os.path.join(folder_b, file_b)
        
        img_a = load_image(path_a)
        img_b = load_image(path_b)
        
        # if flag == "multi":
        #     img_a = cv2.resize(img_a, (256, 256), interpolation=cv2.INTER_AREA) 

        ssim_value = ssim(img_a, img_b, data_range=1)
        psnr_value = psnr(img_a, img_b)
        nmse_value = nmse(img_a, img_b)

        metrics['ssim'].append(ssim_value)
        metrics['psnr'].append(psnr_value)
        metrics['nmse'].append(nmse_value)

    avg_metrics = {key: np.mean(value) for key, value in metrics.items()}
    return avg_metrics

# 使用示例
folder_a = 'MUNIT/testB/'
folder_b = 'outputs/han/test/_00/'
flag = "han"
folder_a = 'result/multi/gt/'
folder_b = 'result/multi/output/'
flag = "multi"
# folder_a = 'MUNIT/testB/'
# folder_b = 'outputs/han/test/_01/'
results = process_folders(folder_a, folder_b, flag)
print(results)
