import torch
import torch.nn as nn

# 1. 准备输入：[batch=1, 输入通道=3, 高=3, 宽=3]
input_data = torch.tensor([
    [
        [[0.,1.,1.], [0.,0.,1.], [0.,0.,0.]],  # 通道1
        [[0.,0.,1.], [1.,0.,1.], [0.,0.,0.]],  # 通道2
        [[1.,0.,1.], [1.,1.,1.], [1.,0.,0.]]   # 通道3
    ]
], dtype=torch.float32)

# 2. 定义2个卷积核（每个3通道，尺寸2x2）
# 卷积核形状：[out_ch, in_ch, k, k]
kernel = torch.tensor([
    # 第1个卷积核（3通道）
    [
        [[1.,0.], [0.,0.]],  # 对应输入通道1
        [[0.,1.], [0.,0.]],  # 对应输入通道2
        [[0.,0.], [1.,0.]]   # 对应输入通道3
    ],
    # 第2个卷积核（3通道）
    [
        [[1.,0.], [0.,0.]],
        [[1.,0.], [0.,0.]],
        [[1.,0.], [0.,0.]]
    ]
], dtype=torch.float32)

# 3. 定义卷积层（输入3通道，输出2通道）
conv_layer = nn.Conv2d(
    in_channels=3,
    out_channels=2,  # 2个卷积核→输出2通道
    kernel_size=2,
    stride=1,
    padding=0,
    bias=False
)
conv_layer.weight.data = kernel  # 加载卷积核权重

# 4. 执行卷积
output = conv_layer(input_data)
print("输出尺寸：", output.shape)  # torch.Size([1, 2, 2, 2])（2通道，2x2特征图）