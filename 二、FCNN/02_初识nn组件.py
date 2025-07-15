"""使用神经网络基本组件完成反向传播"""
import torch
import torch.nn as nn
import torch.optim as optim

# 定义单层网络结构
model = nn.Linear(20, 10)

# 定义损失函数:均方误差= ((y_pred - y)**2).mean()
criterion = nn.MSELoss()

# 定义优化器:整合了梯度清零、参数更新等功能
# 基本参数：
# model.parameters()：模型参数 - 权重和偏置
# lr：学习率
opt = optim.SGD(model.parameters(), lr=0.01)

x = torch.randn(100, 20)
y = torch.randn(100, 10)

epochs = 100

for epoch in range(epochs):
    # 前向传播：根据模型获取预测值
    y_pred = model(x)

    # 计算损失：根据预测值和真实值计算损失
    loss = criterion(y_pred, y)

    # 梯度清零：使用优化器的zero_grad()方法
    opt.zero_grad()

    # 反向传播
    loss.backward()

    # 更新参数：梯度下降，使用优化器的step()方法
    opt.step()

    if epoch % 10 == 0:
        print(f"epoch:[{epoch}/{epochs}], loss={loss.item():.4f}")