import torch
from torchvision import transforms
from torchvision.datasets import MNIST
import random
from model import LeNet

def predict():
    # 定义数据预处理的转换操作
    data_transform = transforms.Compose([
        transforms.Resize(32),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    # 加载MNIST测试集
    test_dataset = MNIST(root='../datasets', train=False,
                         download=False, transform=data_transform)

    # 随机选择一张测试图片
    random_index = random.randint(0, len(test_dataset) - 1)
    test_image, test_label = test_dataset[random_index]
    test_image = test_image.unsqueeze(0)  # 添加一个维度以匹配模型输入

    # 实例化LeNet模型并加载预训练参数
    model = LeNet()
    save_path = '../model/LeNet_MNIST.pth'
    model.load_state_dict(torch.load(save_path))

    # 将模型设置为评估模式
    model.eval()
    with torch.no_grad():
        output = model(test_image)
        predicted_label = output.argmax(dim=1).item()

    print(f'真实结果：{test_label}')
    print(f'预测结果：{predicted_label}')

if __name__ == '__main__':
    predict()