import torch
from torch import nn, optim
from sklearn.datasets import make_circles
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


"""
### 数据准备：生成同心圆数据
make_circles会生成两个同心圆的二分类数据。
- n_samples: 总样本数
- factor: 内外圆半径比例
- noise: 添加的噪声量
- random_state: 随机种子
"""
def build_data():
    x, y = make_circles(
        n_samples=2000,
        factor=0.4,    # 内外圆比例
        noise=0.1,     # 噪声
        random_state=22
    )

    # 将数据转换为PyTorch张量
    x = torch.tensor(x, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.long)  # 目标标签需为long类型（用于CrossEntropyLoss）

    # 数据划分（80%训练集，20%测试集）
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=22
    )

    return x_train, x_test, y_train, y_test


"""
### 构建网络模型：带批量标准化（Batch Normalization）
BN层在每个全连接层后添加，有助于加速训练并提高稳定性。
"""
class NetWithBN(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.fc1 = nn.Linear(in_features, 128)     # 输入层到隐藏层1
        self.bn1 = nn.BatchNorm1d(128)          # BN层，规范化128个特征
        self.relu1 = nn.ReLU()                  # 激活函数

        self.fc2 = nn.Linear(128, 64)    # 隐藏层1到隐藏层2
        self.bn2 = nn.BatchNorm1d(64)           # BN层，规范化64个特征
        self.relu2 = nn.ReLU()                  # 激活函数

        self.fc3 = nn.Linear(64, out_features)  # 输出层（2个类别）

    def forward(self, x):
        x = self.relu1(self.bn1(self.fc1(x)))   # 全连接层 + BN + ReLU
        x = self.relu2(self.bn2(self.fc2(x)))   # 全连接层 + BN + ReLU
        return self.fc3(x)                      # 输出logits（未经过激活）


"""
### 构建网络模型：不使用BN
与NetWithBN结构相同，但去除了BN层。
"""
class NetWithoutBN(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.fc1 = nn.Linear(in_features, 128)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(128, 64)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(64, out_features)

    def forward(self, x):  # 修复拼写错误（原为__forward__）
        x = self.relu1(self.fc1(x))
        x = self.relu2(self.fc2(x))
        return self.fc3(x)


"""
### 训练模型
- 使用CrossEntropyLoss：适用于多类别分类（此处为二分类）
- SGD优化器：随机梯度下降
- model.train()：启用训练模式（如BN和Dropout）
"""
def train(model, x_train, y_train, epochs):
    model.train()  # 启用训练模式
    criterion = nn.CrossEntropyLoss()  # 交叉熵损失函数
    opt = optim.SGD(model.parameters(), lr=0.1)  # 学习率为0.1的SGD
    loss_list = []

    for epoch in range(epochs):
        y_pred = model(x_train)  # 模型预测
        loss = criterion(y_pred, y_train)  # 计算损失
        opt.zero_grad()  # 清除梯度
        loss.backward()  # 反向传播
        opt.step()  # 更新参数

        loss_list.append(loss.item())  # 记录损失值

    return loss_list


"""
### 验证模型
- model.eval()：启用评估模式（关闭BN和Dropout）
- 使用with torch.no_grad()：禁用梯度计算
- 计算准确率
"""
def test(model, x_test, y_test, epochs):
    model.eval()  # 启用评估模式
    acc_list = []

    with torch.no_grad():  # 不计算梯度
        y_pred = model(x_test)  # 模型预测
        _, pred = torch.max(y_pred, dim=1)  # 取最大值索引（类别预测）

        acc = (pred == y_test).sum().item() / len(y_test)  # 计算准确率
        acc_list.append(acc)  # 每次测试记录一次准确率

    return acc_list


"""
### 可视化结果
绘制训练损失和测试准确率对比图
"""
def plot(bn_loss_list, no_bn_loss_list, bn_acc, no_bn_acc):
    fig = plt.figure(figsize=(12, 5))

    # 绘制损失曲线
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.plot(bn_loss_list, 'b', label='BN')
    ax1.plot(no_bn_loss_list, 'r', label='No BN')
    ax1.set_title('Loss Curve')
    ax1.legend()

    # 绘制准确率曲线
    ax2 = fig.add_subplot(1, 2, 2)
    ax2.plot([bn_acc] * 100, 'b', label='BN')  # 测试只运行一次，扩展为100个点
    ax2.plot([no_bn_acc] * 100, 'r', label='No BN')
    ax2.set_title('Accuracy Curve')
    ax2.legend()

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    # 准备数据
    x_train, x_test, y_train, y_test = build_data()

    # 初始化模型
    bn_model = NetWithBN(2, 2)
    no_bn_model = NetWithoutBN(2, 2)

    # 训练模型（100个epoch）
    bn_loss_list = train(bn_model, x_train, y_train, 100)
    no_bn_loss_list = train(no_bn_model, x_train, y_train, 100)

    # 测试模型（仅运行一次）
    bn_acc = test(bn_model, x_test, y_test, 1)[0]
    no_bn_acc = test(no_bn_model, x_test, y_test, 1)[0]

    # 可视化结果
    plot(bn_loss_list, no_bn_loss_list, bn_acc, no_bn_acc)