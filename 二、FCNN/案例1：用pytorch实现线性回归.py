"""用pytorch实现线性回归"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.datasets import make_regression


def build_data(in_features, out_features):
    """生成用于线性回归的合成数据"""
    bias = 14.5  # 设定真实偏置值

    # 使用sklearn的make_regression生成合成数据
    # 参数说明：
    # n_samples=1000: 生成1000个样本
    # n_features=in_features: 每个样本有in_features个特征
    # n_targets=out_features: 每个样本有out_features个目标值
    # coef=True: 返回真实的权重系数
    # bias=bias: 设置偏置值
    # noise=0.1: 添加高斯噪声，标准差为0.1
    # random_state=42: 设置随机种子，确保结果可复现
    x, y, coef = make_regression(
        n_samples=1000,
        n_features=in_features,
        n_targets=out_features,
        coef=True,
        bias=bias,
        noise=0.1,
        random_state=42
    )

    # 将numpy数组转换为PyTorch张量
    x = torch.tensor(x, dtype=torch.float32)
    # view(-1, 1)的作用：
    # 将一维数组重塑为二维数组，其中-1表示该维度的大小由数据自动推断
    # 例如，如果y原本是形状为(1000,)的一维数组，view(-1, 1)后变为(1000, 1)的二维数组
    # 这是因为PyTorch期望目标值的形状为[样本数, 目标维度]
    y = torch.tensor(y, dtype=torch.float32).view(-1, 1)
    coef = torch.tensor(coef, dtype=torch.float32)
    bias = torch.tensor(bias, dtype=torch.float32)

    return x, y, coef, bias


def train():
    # 数据准备
    in_features = 10  # 输入特征维度
    out_features = 1  # 输出维度（目标值数量）
    x, y, coef, bias = build_data(in_features, out_features)

    # 创建数据集和数据加载器
    # TensorDataset将特征和目标值组合成一个数据集
    dataset = TensorDataset(x, y)

    # DataLoader用于批量加载数据，并支持数据打乱和并行加载
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=10,  # 每次加载10个样本
        shuffle=True  # 每个epoch打乱数据顺序
    )

    # 定义网络模型
    # nn.Linear是PyTorch中的全连接层，实现y = xW^T + b
    # 输入维度为in_features，输出维度为out_features
    model = nn.Linear(in_features, out_features)

    # 定义损失函数
    # MSELoss计算预测值和真实值之间的均方误差
    criterion = nn.MSELoss()

    # 优化器
    # SGD（随机梯度下降）用于更新模型参数
    # model.parameters()返回模型的所有可训练参数（权重和偏置）
    opt = optim.SGD(model.parameters(), lr=0.1)

    epochs = 100  # 训练轮数

    for epoch in range(epochs):
        for tx, ty in dataloader:  # 每次迭代加载一个批次的数据
            y_pred = model(tx)  # 前向传播：计算模型预测值

            loss = criterion(y_pred, ty)  # 计算损失值

            opt.zero_grad()  # 梯度清零，防止梯度累积

            loss.backward()  # 反向传播：计算梯度

            opt.step()  # 更新模型参数（权重和偏置）

        if epoch % 10 == 0:
            # 打印每10个epoch的损失值
            print(f"epoch:[{epoch}/{epochs}], loss={loss.item():.4f}")

    print("训练完毕")
    # model.weight是一个Parameter对象，它包含两个属性：
    # - data: 存储参数的实际值（张量）
    # - grad: 存储参数的梯度（也是一个张量）
    # 使用.data可以直接访问参数的张量值
    print(f'真实权重：{coef}，训练权重：{model.weight.data}')
    print(f'真实偏置：{bias}，训练偏置：{model.bias.data}')


if __name__ == '__main__':
    train()