import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt


"""定义CNN模型：使用固定的边缘检测卷积核"""
class EdgeDetectionCNN(nn.Module):
    def __init__(self):
        super(EdgeDetectionCNN, self).__init__()
        # 定义卷积层：in_ch=1，out_ch=2（灰度->水平+垂直边缘）
        self.conv1 = nn.Conv2d(1,2,3, 1,1, bias=False)

        # 定义Sobel算子（固定卷积核，不训练）
        # 水平方向算子：检测垂直边缘
        sobel_x = torch.tensor([[[[-1, 0, 1],
                                  [-2, 0, 2],
                                  [-1, 0, 1]]]], dtype=torch.float32)
        # 垂直方向算子：检测水平边缘
        sobel_y = torch.tensor([[[[-1, -2, -1],
                                  [0, 0, 0],
                                  [1, 2, 1]]]], dtype=torch.float32)

        # 组合两个卷积核
        # 📌dim=0 表示在第0维进行拼接（输出通道维度）
        # 拼接后 shape: [2, 1, 3, 3]，2个卷积核，1个输入通道，3*3
        edge_kernels = torch.cat([sobel_x, sobel_y], dim=0)
        """
        卷积核shape：[out_ch, in_ch, k, k]
        卷积层shape：[in_ch, out_ch, H, W]
        torch.cat()拼接时，dim=0，表示每个卷积核生成一个输出通道
        卷积层是一个"功能层"，从in_ch个通道提取信息，输出out_ch个特征图
        卷积核是一个"工具"，用out_ch个卷积核输出，每个卷积核接收in_ch个通道的信息
        """

        # 📌 nn.Parameter作用：标记为模型可训练参数
        # 手动赋值，但无优化器，权重不会更新（相当于冻结）
        self.conv1.weight = nn.Parameter(edge_kernels)


    def forward(self, x):
        # 执行卷积，提取边缘特征
        edge_features = self.conv1(x)    # [1, 2, H, W] 2个通道，0水平，1垂直

        # 分离水平和垂直边缘特征
        # edge_x, edge_y = torch.split(edge_features, 1, dim=1)
        edge_x = edge_features[:, 0:1, :, :]    # 第0通道
        edge_y = edge_features[:, 1:2, :, :]
        """
        为什么不是[:, 0, :, :] ？ 
        📌 整数索引会降维，切片索引保持维度
        """

        # 计算总边缘特征
        edge_map = torch.sqrt(edge_x**2 + edge_y**2)    # 勾股定理
        return edge_map, edge_x, edge_y


"""图像预处理函数"""
def preprocess_image(image_path):
    # 📌 打开图像并转灰度：PIL是纯二维（H,W）
    image = Image.open(image_path).convert('L')


    # 转张量，归一化
    transform = transforms.Compose([
        transforms.ToTensor(),  # 📌会自动添加通道维度 -> [1, H, W]
        transforms.Normalize((0.5,), (0.5,))
    ])

    # 添加批次维度  [1, 1, H, W]
    image_tensor = transform(image).unsqueeze(0)

    return image_tensor, image


"""结果可视化函数"""
def visualize_edge_map(original, horizontal, vertical, edge_map):
    plt.figure(figsize=(12, 10))

    # 原始图像
    plt.subplot(2, 2, 1)
    plt.imshow(original, cmap='gray')
    plt.title('Original Image')
    plt.axis('off')

    # 水平边缘
    plt.subplot(2, 2, 2)
    plt.imshow(horizontal, cmap='gray')
    plt.title('Horizontal Edge')
    plt.axis('off')

    # 垂直边缘
    plt.subplot(2, 2, 3)
    plt.imshow(vertical, cmap='gray')
    plt.title('Vertical Edge')
    plt.axis('off')

    # 总边缘
    plt.subplot(2, 2, 4)
    plt.imshow(edge_map, cmap='gray')
    plt.title('Total Edge')
    plt.axis('off')

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    # 1.初始化模型
    model = EdgeDetectionCNN()

    # 2.加载和预处理图像
    img_tensor, img = preprocess_image('./datasets/bird.jpg')

    # 3.提取边缘特征
    with torch.no_grad():
        # 📌推理时禁用梯度计算，减少内存占用，加速
        # 本身就没有反向传播，已冻结，但可确保权重不被更新
        edge_map, edge_x, edge_y = model(img_tensor)

    # 4. 转换为numpy并可视化
    edge_map = edge_map.squeeze().numpy()
    edge_x = edge_x.squeeze().numpy()
    edge_y = edge_y.squeeze().numpy()
    visualize_edge_map(img, edge_x, edge_y, edge_map)