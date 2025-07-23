import torch
import torch.nn as nn


"""定义LeNet模型"""
class LeNet(nn.Module):
    def __init__(self):
        super().__init__()
        # 卷积层1：输入1通道（灰度图），输出6通道，5×5卷积核
        # 📌卷积核大小选择——5×5适合提取中等尺度特征，小核（3×3）可叠加提取更细特征
        self.conv1 = nn.Conv2d(1, 6, 5)  # input(1,32,32)   output(6,28,28)
        self.pool1 = nn.MaxPool2d(2, 2, 0)  # output(6,14,14)

        # 卷积层2：输入6通道，输出16通道，5×5卷积核
        self.conv2 = nn.Conv2d(6, 16, 5)  # output(16,10,10)
        self.pool2 = nn.MaxPool2d(2, 2, 0)  # output(16,5,5)

        # 全连接层1：输入为16×5×5（池化后特征图展平），输出120
        self.fc1 = nn.Linear(16 * 5 * 5, 120)    # output(120)
        # 全连接层2：输入120，输出84
        self.fc2 = nn.Linear(120, 84)
        # 输出层：输入84，输出10（对应10个数字类别）
        self.fc3 = nn.Linear(84, 10)

        # 激活函数：ReLU（替代原论文sigmoid，收敛更快）
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.pool1(x)

        x = self.relu(self.conv2(x))
        x = self.pool2(x)

        # 展平特征图：[batch,16,5,5]→[batch,16*5*5]
        # x = x.view(-1, 16 * 5 * 5)
        x = torch.flatten(x, start_dim=1)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)

        return x
