"""
y = ax + b，已知一组x和y，求a和b的值
"""
import torch
import numpy as np

# 数据
x = torch.tensor([1,2,3,4,5], dtype=torch.float32)
y = torch.tensor([3,5,7,9,11], dtype=torch.float32)

# 初始化模型参数
a = torch.tensor(0.0, requires_grad=True)
b = torch.tensor(0.0, requires_grad=True)

epochs = 500    # 训练轮次
lr = 0.01        # 学习率

for epoch in range(epochs):
    # 前向传播
    y_pred = a * x + b

    # 计算损失：MSE
    loss = ((y_pred - y) ** 2).mean()

    # 梯度清零
    if a.grad or b.grad:
        a.grad.zero_()
        b.grad.zero_()

    # 反向传播
    loss.backward()

    # 更新参数
    with torch.no_grad():
        a -= lr * a.grad
        b -= lr * b.grad

    if epoch % 10 ==0:
        print(f"epoch:[{epoch}/{epochs}], loss={loss.item():.4f}, a={a.item():.4f}, b={b.item():.4f}")

print(f"训练完毕，最终 a={a.item():.2f}, b={b.item():.2f}")