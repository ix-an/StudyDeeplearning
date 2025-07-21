"""
使用全连接网络训练和预测MNIST数据集
1.数据准备：数据加载器加载官方MNIST数据集
2.构建网络结构
3.实现训练方法：使用交叉熵损失 + Adam优化器
4.实现验证方法
5.通过测试图片进行预测
"""

import torch
from torch import nn, optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from PIL import Image

"""构建数据集"""
def build_data():

    transform = transforms.Compose([
        transforms.Resize((28,28)),
        transforms.ToTensor()
    ])

    # 训练数据集
    train_dataset = datasets.MNIST(
        root='../datasets/MNIST',
        train=True,
        download=True,
        transform=transform
    )

    # 验证数据集
    test_dataset = datasets.MNIST(
        root='../datasets/MNIST',
        train=False,
        download=True,
        transform=transform
    )

    # 训练数据加载器
    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=64,
        shuffle=True
    )

    # 验证数据加载器
    test_dataloader = DataLoader(
        dataset=test_dataset,
        batch_size=64,
        shuffle=False
    )

    return train_dataloader, test_dataloader


"""构建网络结构"""
class MNISTNet(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        # 输入层
        self.fc1 = nn.Linear(in_features, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.relu1 = nn.ReLU()

        # 隐藏层
        self.fc2 = nn.Linear(128, 64)
        self.bn2 = nn.BatchNorm1d(64)
        self.relu2 = nn.ReLU()

        # 输出层
        self.fc3 = nn.Linear(64, out_features)

    def forward(self, x):
        # 展平输出
        x = x.view(x.size(0), -1)
        x = self.relu1(self.bn1(self.fc1(x)))
        x = self.relu2(self.bn2(self.fc2(x)))
        x = self.fc3(x)
        return x


"""训练模型"""
def train(model, train_dataloader, lr, epochs):
    model.train()
    # 损失函数：交叉熵损失
    criterion = nn.CrossEntropyLoss()
    # 优化器：Adam
    opt = optim.Adam(
        model.parameters(),
        lr=lr,
        betas=(0.9, 0.999),
        weight_decay=0.001
    )

    for epoch in range(epochs):
        correct = 0
        for tx, ty in train_dataloader:
            y_pred = model(tx)              # 前向传播
            loss = criterion(y_pred, ty)    # 计算损失
            opt.zero_grad()                 # 梯度清零
            loss.backward()                 # 反向传播
            opt.step()                      # 更新参数

            # 计算准确率
            _, pred = torch.max(y_pred.data, dim=1)    # 获取预测结果
            correct += (pred == ty).sum().item()

        acc = correct / len(train_dataloader.dataset)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}, Acc: {acc:.4f}")


"""验证模型"""
def test(model, test_dataloader):
    model.eval()

    criterion = nn.CrossEntropyLoss()
    correct = 0

    for vx, vy in test_dataloader:
       with torch.no_grad():
           y_pred = model(vx)
           loss = criterion(y_pred, vy)
           _, pred = torch.max(y_pred.data, dim=1)
           correct += (pred == vy).sum().item()

    acc = correct / len(test_dataloader.dataset)
    print(f'loss:{loss.item():.4f}, acc:{acc:.4f}')

"""模型保存"""
def save_model(model, path):
    torch.save(model.state_dict(), path)

"""模型加载"""
def load_model(path):
    model = MNISTNet(1 * 28 * 28, 10)
    model.load_state_dict(torch.load(path))

    return model

"""模型预测"""
def predict(model, test_path):
    transform = transforms.Compose([
        transforms.Resize((28,28)),
        transforms.ToTensor()
    ])

    img = Image.open(test_path)    # (28,28)
    t_img = transform(img)         # (1,28,28)
    t_img = t_img.unsqueeze(0)     # (1,1,28,28)

    with torch.no_grad():
        y_pred = model(t_img)
        _, pred = torch.max(y_pred.data, dim=1)

    print(f'预测结果：{pred.item()}')


if __name__ == '__main__':
    train_dataloader, test_dataloader = build_data()
    model = MNISTNet(1 * 28 * 28, 10)
    train(model, train_dataloader, lr=0.01, epochs=10)
    test(model, test_dataloader)
    save_model(model, '../models/mnist.pth')
    model = load_model('../models/mnist.pth')
    predict(model, '../datasets/MNIST/test.png')
