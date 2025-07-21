"""函数参数求解：修改版，使用nn组件，如Linear、SGD、MSELoss等"""
import torch
import torch.nn as nn
import torch.optim as optim

# 构建数据
def build_data():
    x = torch.tensor([1, 2, 3, 4, 5], dtype=torch.float32).view(-1, 1)
    y = torch.tensor([3, 5, 7, 9, 11], dtype=torch.float32).view(-1, 1)

    return x, y

# 训练模型
def train_model():
    # 数据准备
    x, y = build_data()

    # 定义网络模型
    # 函数为 y = a*x + b，输入x（1个特征），输出y（1个特征）
    model = nn.Linear(1,1)

    # 定义损失函数：均方误差
    criterion = nn.MSELoss()

    # 定义优化器：SGD
    optimizer = optim.SGD(model.parameters(), lr=0.01)

    epochs = 500

    for epoch in range(epochs):

        y_pred = model(x)
        loss = criterion(y_pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % 50 == 0:
            print(f"epoch:[{epoch}/{epochs}], loss={loss.item():.6f}")

    print(f"训练完毕，最终 a={model.weight.data.item():.2f}, b={model.bias.data.item():.2f}")


if __name__ == "__main__":
    train_model()