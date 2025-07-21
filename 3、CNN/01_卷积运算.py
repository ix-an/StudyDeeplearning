"""完成第一个卷积操作案例，输入层3*3，卷积核2*2，计算卷积结果"""
import torch
import torch.nn as nn

# 定义输入层
input_matrix = torch.tensor([[1.,2.,3.],[4.,5.,6.],[7.,8.,9.]])

# 定义卷积核
kernel = torch.tensor([[0.,1.],[2.,3.]])

# 将输入扩展为四维，即[batch_size, in_channels, height, width]
input_matrix = input_matrix.view(1,1,3,3)

# 定义一个卷积层
conv_layer = nn.Conv2d(
    in_channels=1,    # 输入通道数
    out_channels=1,   # 输出通道数
    kernel_size=2,    # 卷积核大小
    stride=1,         # 步长
    padding=0,        # 是否填充
    bias=False)       # 不使用偏置

# 手动设置卷积层的权重
conv_layer.weight.data = kernel.view(1,1,2,2)

# 执行卷积操作
output_matrix = conv_layer(input_matrix)

print(output_matrix)

# 降维
output_matrix = output_matrix.squeeze().detach()

print(output_matrix)
