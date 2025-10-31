import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

def randomCrop(image, label):
    border = 30
    image_width, image_height = image.size
    crop_win_width = np.random.randint(image_width - border, image_width)
    crop_win_height = np.random.randint(image_height - border, image_height)
    random_region = (
        (image_width - crop_win_width) // 2,
        (image_height - crop_win_height) // 2,
        (image_width + crop_win_width) // 2,
        (image_height + crop_win_height) // 2
    )
    return image.crop(random_region), label.crop(random_region)

# 创建一个全为一的矩阵作为图像和标签
image = Image.fromarray(np.random.randint(0, 256, size=(512, 512), dtype=np.uint8))
label = Image.fromarray(np.random.randint(0, 256, size=(512, 512), dtype=np.uint8))

# 应用随机裁剪
cropped_image, cropped_label = randomCrop(image, label)

# 可视化原始矩阵和裁剪后的矩阵
fig, ax = plt.subplots(2, 2, figsize=(10, 10))

ax[0, 0].imshow(image, cmap='gray')
ax[0, 0].set_title('Original Image')
ax[0, 0].axis('off')

ax[0, 1].imshow(label, cmap='gray')
ax[0, 1].set_title('Original Label')
ax[0, 1].axis('off')

ax[1, 0].imshow(cropped_image, cmap='gray')
ax[1, 0].set_title('Cropped Image')
ax[1, 0].axis('off')

ax[1, 1].imshow(cropped_label, cmap='gray')
ax[1, 1].set_title('Cropped Label')
ax[1, 1].axis('off')

plt.show()
