"""通过梯度下降求函数最小值"""
import torch
import numpy as np
from matplotlib import pyplot as plt

x = torch.tensor(3.0, requires_grad=True)

x_list = []
y_list = []
epochs = 100    # 训练轮次
lr = 0.1    # 学习率

for epoch in range(epochs):
    y = x.pow(2)
    z = y.sum()

    # 梯度清零
    if x.grad is not None:
        x.grad.zero_()

    # 反向传播
    z.backward()

    # 更新参数（梯度下降）
    with torch.no_grad():
        x -= lr * x.grad
    """
    计算图中的叶子结点不允许直接修改值（原始数据），否则会报错
    在通过梯度下降公式进行参数更新时，需要使用with torch.no_grad()来禁止梯度计算
    """

    x_list.append(x.item())
    y_list.append(y.item())

plt.scatter(x_list, y_list)
plt.show()
