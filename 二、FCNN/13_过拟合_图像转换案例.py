import matplotlib.pyplot as plt
import torch
from torchvision import transforms, datasets
from torch.utils.data import DataLoader


def test01():
    # 定义数据增强和归一化
    transform = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),  # 随机水平翻转
            transforms.RandomRotation(10),  # 随机旋转 ±10 度
            transforms.RandomResizedCrop(
                32, scale=(0.8, 1.0)
            ),  # 随机裁剪到 32x32，缩放比例在0.8到1.0之间
            transforms.ColorJitter(
                brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1
            ),  # 随机调整亮度、对比度、饱和度、色调
            transforms.ToTensor(),  # 转换为 Tensor
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),  # 归一化，这是一种常见的经验设置，适用于数据范围 [0, 1]，使其映射到 [-1, 1]
        ]
    )

    # 加载数据集，并应用数据增强
    trainset = datasets.CIFAR10(root="../datasets/MNIST", train=True, download=True, transform=transform)
    dataloader = DataLoader(trainset, batch_size=4, shuffle=False)

    # 获取一个批次的数据
    images, labels = next(iter(dataloader))

    # 还原图片并显示
    plt.figure(figsize=(10, 5))
    for i in range(4):
        # 反归一化：将像素值从 [-1, 1] 还原到 [0, 1]
        img = images[i] / 2 + 0.5

        # 转换为 PIL 图像
        img_pil = transforms.ToPILImage()(img)

        # 显示图片
        plt.subplot(1, 4, i + 1)
        plt.imshow(img_pil)
        plt.axis('off')
        plt.title(f'Label: {labels[i]}')

    plt.show()


if __name__ == "__main__":
    test01()