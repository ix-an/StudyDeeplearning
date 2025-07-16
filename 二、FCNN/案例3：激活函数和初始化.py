import torch
import torch.nn as nn
from torch.nn import functional as F


"""网络模型如何与激活函数和参数初始化结合使用"""
class MyNet(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.fc1 = nn.Linear(in_features, 128)
        nn.init.kaiming_uniform(self.fc1.weight)
        self.relu1 = nn.ReLU()

        self.fc2 = nn.Linear(128, 64)
        nn.init.kaiming_uniform(self.fc2.weight)
        self.relu2 = nn.ReLU()

        self.fc3 = nn.Linear(64, out_features)
        nn.init.kaiming_uniform(self.fc3.weight)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu1(self.fc1(x))
        x = self.relu2(self.fc2(x))
        x = self.sigmoid(self.fc3(x))
        return x

"""也可以使用functional API"""
class MyNet2(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.fc1 = nn.Linear(in_features, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, out_features)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def test_net():
    model1 = MyNet(10, 2)

    print(model1)


if __name__ == '__main__':
    test_net()