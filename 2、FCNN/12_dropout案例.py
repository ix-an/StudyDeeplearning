import torch
from torch import nn
from PIL import Image
from torchvision import transforms
import os

from matplotlib import pyplot as plt

torch.manual_seed(42)


def load_img(path, resize=(224, 224)):
    pil_img = Image.open(path).convert('RGB')
    print("Original image size:", pil_img.size)  # 打印原始尺寸
    transform = transforms.Compose([
        transforms.Resize(resize),
        transforms.ToTensor()  # 转换为Tensor并自动归一化到[0,1]
    ])
    return transform(pil_img)  # 返回[C,H,W]格式的tensor


if __name__ == '__main__':
    dirpath = os.path.dirname(__file__)
    path = '../datasets/images/100.jpg'
    # 加载图像 (已经是[0,1]范围的Tensor)
    trans_img = load_img(path)

    # 添加batch维度 [1, C, H, W]，因为Dropout默认需要4D输入
    trans_img = trans_img.unsqueeze(0)

    # 创建Dropout层
    dropout = nn.Dropout2d(p=0.2)

    drop_img = dropout(trans_img)

    # 移除batch维度并转换为[H,W,C]格式供matplotlib显示
    trans_img = trans_img.squeeze(0).permute(1, 2, 0).numpy()
    drop_img = drop_img.squeeze(0).permute(1, 2, 0).numpy()

    # 确保数据在[0,1]范围内
    drop_img = drop_img.clip(0, 1)

    # 显示图像
    fig = plt.figure(figsize=(10, 5))

    ax1 = fig.add_subplot(1, 2, 1)
    ax1.imshow(trans_img)

    ax2 = fig.add_subplot(1, 2, 2)
    ax2.imshow(drop_img)

    plt.show()