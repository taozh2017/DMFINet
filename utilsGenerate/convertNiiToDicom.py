import SimpleITK as sitk
import numpy as np
import os

def nii_to_dicom(nii_file, output_dicom_dir):
    os.makedirs(output_dicom_dir, exist_ok=True)
    
    # 读取 NIfTI 文件
    image = sitk.ReadImage(nii_file)
    
    # 将图像数据转换为 NumPy 数组
    array = sitk.GetArrayFromImage(image)

    array = array * 2000
    
    # 打印原始数组的形状
    print(f"Original array shape: {array.shape}")
    
    # 转换形状为 (1, 256, 256)
    if array.ndim == 3 and array.shape[0] == 1:
        # 将 (256, 1, 256) 转换为 (1, 256, 256)
        array = np.squeeze(array, axis=0)
    elif array.ndim == 3 and array.shape[1] == 1:
        # 将 (256, 1, 256) 转换为 (1, 256, 256)
        array = np.squeeze(array, axis=1)
    else:
        raise ValueError("Unexpected array shape.")
    
    # 确保形状正确
    array = np.expand_dims(array, axis=0)  # 增加通道维度，形状变为 (1, 256, 256)
    
    # 打印新数组的形状
    print(f"Reshaped array shape: {array.shape}")
    
    # 将 NumPy 数组转换回 SimpleITK 图像
    image = sitk.GetImageFromArray(array)
    
    # 确保图像数据类型为 uint16
    image = sitk.Cast(image, sitk.sitkUInt16)
    
    # 获取文件名，不包括扩展名
    base_filename = os.path.basename(nii_file).replace('.nii.gz', '')
    
    # 将图像写入 DICOM 文件
    dicom_file = os.path.join(output_dicom_dir, f"{base_filename}.dcm")
    
    # 使用 SimpleITK 保存为 DICOM 文件
    sitk.WriteImage(image, dicom_file)
    
    print(f"NIfTI file {nii_file} converted to DICOM file {dicom_file}.")

# 示例调用
result_dir = 'result/multi/'
output_dicom_dir = 'result/multi/'

nii_file = os.path.join(result_dir, "9999nii.gz.nii.gz")  # 确保文件扩展名正确
nii_to_dicom(nii_file, output_dicom_dir)
