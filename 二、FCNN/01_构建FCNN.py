import torch
from torch import nn    # nn是神经网络的缩写

"""创建FCNN步骤：
1.需要继承nn.Module抽象类
2.实现__init__()方法，在此定义网络结构
3.实现forward()方法，在此定义网络的前向传播
"""
# 定义网络结构，要继承nn.Module
class MyNet(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()    # 父类的初始化

        self.fc1 = nn.Linear(in_features, 128)  # (输入数据的维度，输出数据的维度)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, out_features)

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)

        return x

def test01():
    in_features = 50
    out_features = 6

    model = MyNet(in_features, out_features)

    print(model)


"""使用nn.Sequential()快速创建神经网络
它可以自动实现forward()方法，默认从上到下依次执行网络结构
注意：在Sequential()中，定义网络结构时，需要按照顺序定义
     即：输入数据 -> 第一层 -> 第二层 -> ... -> 输出数据
"""
def test02():
    in_features = 50
    out_features = 6
    model = nn.Sequential(
        nn.Linear(in_features, 128),
        nn.Linear(128, 64),
        nn.Linear(64, out_features)
    )
    print(model)

"""创建单层网络结构
直接使用nn.Linear()创建即可
"""
def test03():
    in_features = 50
    out_features = 1

    model = nn.Linear(in_features, out_features)
    print(model)


if __name__ == '__main__':
    # test01()
    # test02()
    test03()


