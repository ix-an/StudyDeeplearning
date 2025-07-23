import torch
import torch.nn as nn
from torchvision import transforms, datasets
from torch import optim
from torch.utils.data import DataLoader
from model import LeNet
import matplotlib.pyplot as plt


def main():
    # 使用GPU进行计算
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    transform = transforms.Compose([
        transforms.Resize(32),  # 符合LeNet输入要求
        transforms.ToTensor(),
        # 📌注意归一化便捷写法，括号内有逗号
        transforms.Normalize((0.5,), (0.5,))
    ])

    """
    MNIST数据集
       数据集:70000张，手写数字集合
       训练集:60,000 张图像，测试集:10,000 张图像
       每个图像大小:28×28像素，单通道（灰度图）

    num_workers: 
       小数据集：对于小型数据集，可以设置为 0 或 1，因为并行加载的开销可能大于加速效果。
       大型数据集：对于大型数据集，尤其是需要进行复杂数据处理时，设置更高的 num_workers 会有明显的性能提升。
       num_workers 的值可以根据 CPU 核心数来选择，比如设置为 4、8 或更高。
    """

    # 加载数据集
    train_set = datasets.MNIST(root='../datasets',train=True,
                               download=False,transform=transform)
    train_loader = DataLoader(dataset=train_set,batch_size=64,
                              shuffle=True,num_workers=0)

    test_set = datasets.MNIST(root='../datasets',train=False,
                              download=False,transform=transform)
    test_loader = DataLoader(dataset=test_set,batch_size=64,
                             shuffle=False,num_workers=0)

    # 构建一个迭代器，用于迭代出测试数据
    # 因为MNIST测试集只有10000张，且batch_size设置了10000
    # 那么可以直接迭代一次，获取测试数据
    dataiter = iter(test_loader)
    test_images, test_labels = next(dataiter)
    test_images = test_images.to(device)
    test_labels = test_labels.to(device)


    """
    进行训练：固定写法
    """
    # 实例化模型
    net = LeNet()
    net = net.to(device)

    # 损失函数
    loss = nn.CrossEntropyLoss()
    loss = loss.to(device)

    # 优化器
    opt = optim.Adam(net.parameters(),lr=0.01)

    # 存储每一轮的平均损失和准确率
    train_loss_list = []
    train_acc_list = []

    # 训练轮数
    for epoch in range(5):
        running_loss = 0.0
        for step, data in enumerate(train_loader, start=0):
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)

            # 前向传播
            outputs = net(inputs)
            loss_value = loss(outputs, labels)
            # 反向传播和优化
            opt.zero_grad()
            loss_value.backward()
            opt.step()

            # 统计指标
            running_loss += loss_value.item()
            if step % 100 == 99:
                with torch.no_grad():
                    outputs = net(test_images)
                    # dim=1按行找最大值的索引
                    predict_y = torch.argmax(outputs, dim=1)
                    # test_labels.size(0)返回样本数量，test_labels.size()返回张量形状
                    acc = torch.eq(predict_y, test_labels).sum() / test_labels.size(0)
                    print(f'Epoch:{epoch+1} | Batch:{step+1} | Loss:{running_loss/100:.4f} | Accuracy:{acc:.3f}')
                    running_loss = 0.0


    print('Finished Training')
    # 保存模型参数
    save_path = '../model/LeNet_MNIST.pth'
    torch.save(net.state_dict(), save_path)

if __name__ == '__main__':
    main()




