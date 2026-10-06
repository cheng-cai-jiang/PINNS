from PIL import Image
import numpy as np


def png_to_npz_fixed_size(png_path, npz_path, size=(128, 128)):
    """
    将PNG图像转换为指定大小（默认128x128）的NPZ文件。

    参数:
    - png_path: PNG图像文件的路径。
    - npz_path: 要保存的NPZ文件的路径。
    - size: 目标图像大小，默认为(128, 128)。
    """
    # 打开图像并调整大小
    with Image.open(png_path).convert('L') as img:
        img_resized = img.resize(size, Image.Resampling.LANCZOS)

        # 将图像转换为NumPy数组
        img_array = np.array(img_resized)

    # 创建一个字典来存储数据
    data = {'image': img_array}

    # 保存为npz文件
    np.savez_compressed(npz_path, **data)

    print(f"Image saved to {npz_path} with shape {img_array.shape}")


# 示例使用
png_file = "HG10.png"  # 输入PNG文件路径
npz_file = "HG10.npz"  # 输出NPZ文件路径

png_to_npz_fixed_size(png_file, npz_file)