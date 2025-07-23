import torch
import torch.nn as nn



class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()

        # self.conv1 = nn.Conv2d(3, 8, 3, 1, 1)
        # self.relu1 = nn.ReLU()
        # self.pool1 = nn.MaxPool2d(2, 2)
        #
        # self.conv2 = nn.Conv2d(8, 16, 3, 1, 1)
        # self.relu2 = nn.ReLU()
        # self.pool2 = nn.MaxPool2d(2, 2)

        # 使用 nn.Sequential()，构建序列化网络结构
        self.features = nn.Sequential(
            nn.Conv2d(3, 8, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(8, 16, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )


    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        return x


if __name__ == '__main__':
    model = SimpleCNN()
    print("-----输出模型-----")
    print(model)

