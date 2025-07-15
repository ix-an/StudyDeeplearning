import torch
import torch.nn as nn
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


if __name__ == '__main__':
    # load_csv_data()
    load_image_data()