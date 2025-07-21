"""
使用FCNN对CIFAR10数据集进行训练和验证
CIFAR10数据集：包含10个类别，每个类别6000张32*32的彩色图片
"""
import torch
from torch import nn, optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from PIL import Image
import time

torch.manual_seed(22)

"""数据准备：返回CIFAR10测试集和训练集的数据加载器"""
def build_data():
    # 数据预处理
    train_transform = transforms.Compose([
        transforms.RandomRotation(15),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.4914, 0.4822, 0.4465],  # 三通道均值
            std=[0.2023, 0.1994, 0.2010]  # 三通道标准差
        )
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.4914, 0.4822, 0.4465],  # 三通道均值
            std=[0.2023, 0.1994, 0.2010]  # 三通道标准差
        )
    ])

    # CIFAR10 训练集
    train_dataset = datasets.CIFAR10(
        root='../datasets/cifar10',
        train=True,
        download=False,
        transform=train_transform
    )

    # CIFAR10 测试集
    test_dataset = datasets.CIFAR10(
        root='../datasets/cifar10',
        train=False,
        download=False,
        transform=test_transform
    )

    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=64,  # 增大批大小以加快训练
        shuffle=True,
        num_workers=0,
        pin_memory=True  # 加速GPU数据传输
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=64,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )

    return train_loader, test_loader

"""构建网络模型"""
class CIFAR10Net(nn.Module):
    def __init__(self, in_features, out_features):
        super(CIFAR10Net, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(in_features, 2048),  # 再增加宽度
            nn.BatchNorm1d(2048),
            nn.LeakyReLU(0.1),  # 使用LeakyReLU
            nn.Dropout(0.4),  # dropout调小

            nn.Linear(2048, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.3),

            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.2),

            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.1),

            nn.Linear(256, out_features)
        )
        # 初始化权重
        self._initialize_weights()

    def _initialize_weights(self):
        """新增：初始化网络权重"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)


    def forward(self, x):
        # 将数据展平为二位张量 (batch_size, 32*32*3)
        x = x.view(x.size(0), -1)
        return self.network(x)


"""模型训练"""
def train(model, lr, epochs, device):

    # 先保证移动设备
    model.to(device)

    # 准备数据
    train_loader, test_loader = build_data()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=lr,
        betas=(0.9, 0.999),
        weight_decay=0.0001
    )
    """学习率调度器
    监控验证集准确率，当准确率不再提高时，降低学习率
    """
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='max',
        factor=0.5,
        patience=3,
        verbose=True
    )

    # 初始化早停变量
    best_acc = 0
    early_stop_counter = 0

    print('开始训练')
    print("Epoch | Train Loss | Train Acc | Test Acc | LR")
    print("------------------------------------------------")

    for epoch in range(epochs):
        start_time = time.time()
        model.train()
        train_loss = 0
        correct = 0
        total = 0

        for tx, ty in train_loader:
            tx, ty = tx.to(device), ty.to(device)

            y_pred = model(tx)
            loss = criterion(y_pred, ty)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # 统计指标
            train_loss += loss.item()
            _, pred = y_pred.max(1)
            total += ty.size(0)
            correct += pred.eq(ty).sum().item()

        # 计算训练集准确率
        train_acc = 100. * correct / total
        avg_loss = train_loss / len(train_loader)

        # 测试阶段
        test_acc = test(model, test_loader, device)

        # 更新学习率
        scheduler.step(test_acc)
        current_lr = optimizer.param_groups[0]['lr']

        # 打印每轮结果
        epoch_time = time.time() - start_time
        print(f"{epoch+1:5d} | {avg_loss:10.4f} | {train_acc:8.2f}% | {test_acc:7.2f}% | {current_lr:.6f}")

        # 早停
        if test_acc > best_acc:
            best_acc = test_acc
            early_stop_counter = 0
        else:
            early_stop_counter += 1

        if early_stop_counter >= 10:
            print(f"早停法出发，在第 {epoch}轮终止训练。")
            print(f'此时最佳准确率：{best_acc:.2f}%')
            break

    print("训练完成")
    print(f'此时最佳准确率：{best_acc:.2f}%')
    return best_acc

"""模型验证"""
def test(model, test_loader, device):
    model.eval()
    model.to(device)  # 将模型移动到GPU
    correct, total = 0, 0

    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    return 100. * correct / total

"""模型保存"""
def save_model(model, path):
    torch.save(model.state_dict(), path)
    print(f'模型已保存到{path}')

"""模型加载"""
def load_model(path, device):  #数
    model = CIFAR10Net(32*32*3, 10)
    model.load_state_dict(torch.load(path, map_location=device))
    model.to(device)  # 将模型移动到GPU
    return model

if __name__ == '__main__':
    # 确定设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'当前设备：{device}')
    # 创建网络模型
    model = CIFAR10Net(32 * 32 * 3, 10)
    model.to(device)
    # 训练模型
    train(model, 0.001, 100, device)
    # 保存模型
    save_model(model, '../models/cifar10.pt')
    # # 加载模型
    # model = load_model('../models/cifar10.pt', device)

    # # 预测新数据
    # predict(model, '../datasets/cifar10/test/0_airplane.png', device)
    # predict(model, '../datasets/cifar10/test/1_automobile.png', device)