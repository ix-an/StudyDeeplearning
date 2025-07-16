import torch
import torch.nn as nn

"""随机初始化"""
def test01():
    model = nn.Linear(8, 1)
    # 默认是Kaiming初始化
    print(model.weight)
    """均匀分布初始化"""
    # nn.init.uniform_(model.weight)
    # print(model.weight)
    """正态分布初始化
    参数：
        mean：均值
        std：标准差
    """
    nn.init.normal_(model.weight, mean=0, std=0.01)
    print(model.weight)


"""xavier初始化
原理：前向传播的方差保持一致，反向传播的方差保持一致
"""
def test02():
    model = nn.Linear(8, 1)

    # 均匀分布初始化
    # nn.init.xavier_uniform_(model.weight)
    # print(model.weight)

    # 正态分布初始化
    nn.init.xavier_normal_(model.weight)
    print(model.weight)

"""He初始化"""
def test03():
    model = nn.Linear(8, 1)

    # 均匀分布初始化
    # nn.init.xavier_uniform_(model.weight，mode="fan_in",nonlinearity="relu")
    # print(model.weight)

    # 正态分布初始化
    nn.init.kaiming_normal_(model.weight, mode='fan_in', nonlinearity='relu')
    print(model.weight)


if __name__ == "__main__":
    # test01()
    test02()
