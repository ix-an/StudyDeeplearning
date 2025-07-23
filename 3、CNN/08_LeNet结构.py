import torch
import torch.nn as nn
import torch.nn.functional as F

class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        # 定义卷积层
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1)

        # 定义全连接层
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

        # 定义激活函数
        self.relu = nn.ReLU()

    def forward(self, x):
        # 卷积层 -> 激活函数 -> 池化层
        x = self.relu(self.conv1(x))
        x = F.avg_pool2d(x, kernel_size=2, stride=2)
        x = self.relu(self.conv2(x))
        x = F.avg_pool2d(x, kernel_size=2, stride=2)

        # 展平特征图
        x = torch.flatten(input=x, start_dim=1)    # [batch_size,16,5,5] -> [batch_size,16*5*5]
        # nn.Linear() 只接受二维输入 [batch_size, in_features]

        # 全连接层
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)

        return x

# 创建模型实例
model = LeNet()

# 打印模型结构
print(model)