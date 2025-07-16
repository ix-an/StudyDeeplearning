import torch
from torch.utils.data import TensorDataset, DataLoader
import pandas as pd

def build_csv_data(file_path):
    df = pd.read_csv(file_path)
    df.drop(['学号','姓名'], axis=1, inplace=True)

    samples = df.iloc[..., :-1]
    labels = df.iloc[..., -1]

    samples = torch.tensor(samples.values, dtype=torch.float32)
    labels = torch.tensor(labels.values, dtype=torch.float32).view(-1, 1)

    return samples, labels

def load_csv_data():
    csv_path = '../datasets/大数据答辩成绩表.csv'
    samples, labels = build_csv_data(csv_path)

    dataset = TensorDataset(samples, labels)

    dataloader = DataLoader(
        dataset=dataset,
        batch_size=1,
        shuffle=True
    )

    for sample, label in dataloader:
        print(sample)
        print(label)
        break

"""加载图片数据集"""
from torchvision import datasets, transforms
def load_image_data():
    path = '../datasets/animals'

    transform = transforms.Compose([
        # 缩放图片,统一尺寸
        transforms.Resize((224,224)),
        # 把PIL图片或numpy数组转换成tensor
        transforms.ToTensor()
    ])

    dataset = datasets.ImageFolder(
        root=path,
        transform=transform
    )

    dataloader = DataLoader(
        dataset=dataset,
        batch_size=4,
        shuffle=True
    )

    for x, y in dataloader:
        print(x.shape)
        print(y.shape)
        break


"""加载官方数据集"""
def load_MNIST_data():
    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    """MNIST构造函数:0-9的手写数字图片，每张图片的尺寸为28*28
    root: 存储数据集的本地路径
    train: True表示训练集,False表示测试集
    transform: 数据转换
    download: True表示下载数据集,False表示不下载
    """
    dataset = datasets.MNIST(
        root='../datasets/MNIST',
        train=True,
        transform=transform,
        download=True
    )

    dataloader = DataLoader(
        dataset=dataset,
        batch_size=4,
        shuffle=True
    )

    for x, y in dataloader:
        print(x.shape)
        print(y.shape)
        break








if __name__ == '__main__':
    # load_csv_data()
    # load_image_data()
    load_MNIST_data()