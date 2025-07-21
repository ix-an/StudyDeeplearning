import torch
import torch.nn as nn
import numpy as np

# 创建一个大小为 5*5 的单通道图像
# [批处理大小，通道数，高，宽]
# 第一种方式：torch.tensor()创建张量后，用tensor.view()方法将其转为4维
# 第二种方式：使用torch.randn()等方法，直接创建4维张量
input_data = torch.randn(1, 1, 5, 5)

# 第三种方式：使用numpy构造
input_array_data = np.array(
    [[[[0.0, 2.0, 4.0, 1.0, 0.0],
       [3.0, 1.0, 1.0, 0.0, 1.0],
       [2.0, 4.0, 1.0, 0.0, 1.0],
       [2.0, 0.0, 5.0, 2.0, 2.0],
       [0.0, 1.0, 3.0, 2.0, 1.0]]]]
)
# 将numpy数组转为张量
input_array_data = torch.from_numpy(input_array_data).float()


# 创建卷积层
conv_layer = nn.Conv2d(
    in_channels=1,
    out_channels=1,
    kernel_size=3,
    stride=1,
    padding=1
)

# 对输入数据进行卷积操作
output_data = conv_layer(input_data)

# 计算输出尺寸 (W-F+2*P) / S + 1
print(output_data.shape)
