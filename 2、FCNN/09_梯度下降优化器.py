import torch
from torch import nn
import torch.optim as optim
from torch.utils.data import TensorDataset,DataLoader


def test():
    model = nn.Linear(10, 5)
    x = torch.randn(1000, 10)
    y = torch.randn(1000, 5)

    criterion = nn.MSELoss()

    """Adgrad:自适应学习率优化器
    原理：历史梯度平方法和作为学习率的分母，动态调整学习率
    优点：自适应动态调整学习率
    缺点：随着训练时间增加，历史梯度平方和越来越大，导致学习率越来越小，可能会停止参数更新
    eps：避免分母0
    """
    # opt = optim.Adagrad(model.parameters(), lr=0.10, eps=1e-8)

    """RMSprop:自适应学习率优化器
    原理：使用指数加权平均对历史梯度平方求和，求平方和作为分母调整学习率
    优点：缓解历史梯度平方和快速变大，使学习率衰减更加平稳
    缺点：需要调整alpha和lr参数，找到最优值
    """
    # opt = optim.RMSprop(model.parameters(), lr=0.10, alpha=0.9, eps=1e-8)

    """Adam:自适应优化器
    优点：结合了动量和RMSporop，既优化了梯度，又能动态调整学习率
    缺点：参数敏感，需要调参
    betas参数：一个元组，第一个参数是一阶动量的系数0.9，第二个是二阶动量的系数0.999
        两个系数值是经验值
    """
    opt = optim.Adam(model.parameters(), lr=0.1, betas=(0.9, 0.999), eps=1e-8)


    for epoch in range(50):
        y_pred = model(x)
        loss = criterion(y_pred, y)

        opt.zero_grad()
        loss.backward()
        opt.step()

        print(f"loss:{loss.item()}")


if __name__ == '__main__':
    test()
