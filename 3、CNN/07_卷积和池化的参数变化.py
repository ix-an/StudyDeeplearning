import torch
import torch.nn as nn

# 定义张量x，它的尺寸是1×1×28×28
# 表示了1个，单通道，32×32大小的数据
x = torch.zeros([1, 1, 32, 32])
# 定义一个输入通道是1，输出通道是6，卷积核大小是5x5的卷积层
conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5)
# 将x，输入至conv，计算出结果c
c1 = conv1(x)
# 打印结果尺寸程序输出：
print(c1.shape)  # (32 -5 + 2*0) / 1 + 1 = 28

# 定义最大池化层
pool = nn.MaxPool2d(2)
# 将卷积层计算得到的特征图c，输入至pool
s1 = pool(c1)
# 输出s的尺寸
print(s1.shape)    # 最大池化，size2步长1，缩小一半 14

# 定义第二个输入通道是6，输出通道是16，卷积核大小是5x5的卷积层
conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5)
# 将x，输入至conv，计算出结果c
c2 = conv2(s1)
# 打印结果尺寸程序输出：
print(c2.shape)  # (14-5+0)/1 +1 = 10

s2 = pool(c2)
# 输出s的尺寸
print(s2.shape)  # 再缩小一半 5
