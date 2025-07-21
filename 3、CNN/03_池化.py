import torch
import torch.nn as nn
import numpy as np

# 1.定义输入张量
matrix_np = np.float32([[[[1.0, 0.0, 0.0, 0.0, 1.0],
                          [0.0, 1.0, 0.0, 1.0, 0.0],
                          [0.0, 0.0, 1.0, 0.0, 0.0],
                          [0.0, 1.0, 0.0, 1.0, 0.0],
                          [1.0, 0.0, 0.0, 0.0, 1.0]]]])
# 转为张量
input_data = torch.from_numpy(matrix_np).float()
# print(input_data.size())    # torch.Size([1, 1, 5, 5])

# 2. 定义卷积核
kernel = torch.tensor([
    [0., 0., 1.],
    [0., 1., 0.],
    [1., 0., 0.]
], dtype=torch.float32)
# print(kernel.size())    # torch.Size([3, 3])

# 3. 创建卷积层
conv_layer = nn.Conv2d(
    in_channels=1,
    out_channels=1,
    kernel_size=3,
    stride=1,
    padding=1,
    bias=False
)
# 4. 设置卷积核参数
conv_layer.weight.data = kernel.view(1, 1, 3, 3)

# 5. 执行卷积操作
output_data = conv_layer(input_data)

# 输出结果
print("卷积后尺寸：", output_data.size())
print("卷积结果：", output_data, sep="\n")

# 6. 定义池化层
# 最大池化
max_pool_layer = nn.MaxPool2d(kernel_size=2, stride=3)
# 平均池化
avg_pool_layer = nn.AvgPool2d(kernel_size=2, stride=3)

# 7. 应用池化操作
max_pooled = max_pool_layer(output_data)
avg_pooled = avg_pool_layer(output_data)

# 输出结果
print("最大池化结果：", max_pooled, sep="\n")
print("平均池化结果：", avg_pooled, sep="\n")





