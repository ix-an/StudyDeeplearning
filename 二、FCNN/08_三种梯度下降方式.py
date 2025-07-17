import torch
from torch.utils.data import TensorDataset,DataLoader
from torch import nn, optim

def test():
    # 创建一个线性层
    model = nn.Linear(10, 5)
    # 数据准备
    x = torch.randn(10000, 10)
    y = torch.randn(10000, 5)
    # 定义数据集
    dataset = TensorDataset(x, y)
    """通过定义数据加载器的batch_size，实现三种梯度下降方式"""
    # 定义数据加载器

    # 批量梯度下降 BGD
    # dataloader = DataLoader(
    #     dataset=dataset,
    #     batch_size = len(dataset),
    #     shuffle=True
    # )

    # 随机梯度下降 SGD
    # dataloader = DataLoader(
    #     dataset=dataset,
    #     batch_size = 1,
    #     shuffle=True
    # )

    # 小批量梯度下降 Mini-BGD（MBGD）
    dataloader = DataLoader(
        dataset=dataset,
        batch_size = 32,
        shuffle=True
    )

    # 定义损失函数
    criterion = nn.MSELoss()
    # 定义优化器
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9)    # 添加了动量项
    """momentum动量项：根据历史梯度增加"惯性"
    参数值：动量系数，一般取0.9
    """

    epochs = 20

    for epoch in range(epochs):
        for tx, ty in dataloader:
            # 获取预测值
            y_pred = model(tx)
            # 获取损失
            loss = criterion(y_pred, ty)

            # 梯度清零
            optimizer.zero_grad()
            # 反向传播
            loss.backward()
            # 参数更新
            optimizer.step()

        print(f'epoch:{epoch}, loss={loss.item():.6f}')


if __name__ == '__main__':
    test()
