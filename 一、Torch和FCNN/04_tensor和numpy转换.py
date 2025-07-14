"""
tensor和numpy的转换，分别有浅拷贝和深拷贝
浅拷贝：tensor对象和numpy数组共享内存，修改其中一个，另一个也会改变
深拷贝：tensor对象和numpy数组不共享内存，修改其中一个，另一个不会改变
"""
import torch
import numpy as np

def tensor2numpy():
    """tensor转numpy：numpy()方法浅拷贝
        额外使用copy()方法深拷贝
    """
    a = torch.tensor([[1,2,3],[4,5,6]])
    print(a)    # tensor对象

    # 浅拷贝
    b = a.numpy()
    print(b)    # numpy数组

    # 深拷贝
    c = a.numpy().copy()
    print(c)    # numpy数组

def numpy2tensor():
    """numpy转tensor：
    浅拷贝：torch.from_numpy()
    深拷贝：torch.tensor()，参数为numpy数组"""
    a = np.array([[1,2,3],[4,5,6]])
    print(a)    # numpy数组

    # 浅拷贝
    b = torch.from_numpy(a)
    print(b)    # tensor对象

    # 深拷贝
    c = torch.tensor(a)
    print(c)    # tensor对象

if __name__ == '__main__':
    tensor2numpy()
    print("----------")
    numpy2tensor()