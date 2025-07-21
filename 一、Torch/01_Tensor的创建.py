"""
两种方式：
1. 使用tensor()方法创建张量，可以创建标量、一维、二维、n维都可以
2. 使用Tensor()构造函数
3. 其它方法：类似numpy的方法创建张量
    torch.from_numpy() 从numpy数组创建张量
    torch.zeros() 创建全0张量
    torch.ones() 创建全1张量
    torch.arange() 创建等差张量
    torch.linspace() 创建等间隔张量
    torch.logspace() 创建等比张量
    torch.eye() 创建单位矩阵
    torch.randn() 创建正态分布张量
"""
import torch
import numpy as np


def test_01():
    """1. 使用tensor()方法"""
    t1 = torch.tensor([1, 2, 3])  # 用list创建了一个向量
    print(t1)    # tensor([1, 2, 3])
    print(t1.dtype, t1.shape, t1.size(), t1.device) # torch.int64 torch.Size([3]) torch.Size([3]) cpu
    # .shape数量 和 .size()方法效果一样，但是更常用 .size()方法去打印形状
    # dtype:获取张量的数据类型，如果在创建张量的时候没有指定数据类型，则自动根据输入的数据类型判断
    # device:指定张量运算的设备，cpu、cuda, 默认cpu

    t2 = torch.tensor(17)  # 用标量创建了一个标量（0维张量）
    t3 = torch.tensor(np.random.randint(1, 101, (3,4)), dtype=torch.int8)  # 用numpy随机一个数组创建张量
    print(t2,"\n",t3)


def test_02():
    """2. 使用Tensor()构造函数"""
    t1 = torch.Tensor([1, 2, 3])
    print(t1, t1.dtype)  # tensor([1., 2., 3.]) torch.float32
    # 强制转换为torch.float32
    # 没有dtype和device属性
    # tensor()创建张量更灵活，使用更多




if __name__ == '__main__':
    # test_01()
    test_02()