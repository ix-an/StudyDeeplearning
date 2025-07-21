import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# 设置随机种子以保证可重复性
torch.manual_seed(42)

# 生成随机数据
n_samples = 100
n_features = 20
X = torch.randn(n_samples, n_features)  # 输入数据
y = torch.randn(n_samples, 1)  # 目标值


# 定义一个简单的全连接神经网络
class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(n_features, 50)
        self.fc2 = nn.Linear(50, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return self.fc2(x)


# 训练函数
def train_model(use_l2=False, weight_decay=0.01, n_epochs=100):
    # 初始化模型
    model = SimpleNet()
    criterion = nn.MSELoss()  # 损失函数（均方误差）

    # 选择优化器
    if use_l2:
        optimizer = optim.SGD(model.parameters(), lr=0.01, weight_decay=weight_decay)  # 使用 L2 正则化
    else:
        optimizer = optim.SGD(model.parameters(), lr=0.01)  # 不使用 L2 正则化

    # 记录训练损失
    train_losses = []

    # 训练过程
    for epoch in range(n_epochs):
        optimizer.zero_grad()  # 清空梯度
        outputs = model(X)  # 前向传播
        loss = criterion(outputs, y)  # 计算损失
        loss.backward()  # 反向传播
        optimizer.step()  # 更新参数

        train_losses.append(loss.item())  # 记录损失

        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch + 1}/{n_epochs}], Loss: {loss.item():.4f}')

    return train_losses


# 训练并比较两种模型
train_losses_no_l2 = train_model(use_l2=False)  # 不使用 L2 正则化
train_losses_with_l2 = train_model(use_l2=True, weight_decay=0.01)  # 使用 L2 正则化

# 绘制训练损失曲线
plt.plot(train_losses_no_l2, label='Without L2 Regularization')
plt.plot(train_losses_with_l2, label='With L2 Regularization')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss: L2 Regularization vs No Regularization')
plt.legend()
plt.show()