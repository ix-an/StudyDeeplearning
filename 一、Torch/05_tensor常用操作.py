import torch
import numpy as np

"""1.获取元素值: .item()"""
def get_element():

    t1 = torch.tensor([[17]])
    print(t1.item())

    t2 = torch.tensor([1,2,3])
    print(t2[0].item())
    """
    item()获取元素和tensor的维度没有关系
    只要tensor里只有一个元素,就可以取值成Python标量
    """

"""2.tensor运算函数：如果函数名后带有下划线_，则表示原地修改方法"""

"""3. 点乘（阿达玛积）和Tensor乘法（矩阵乘法）"""
def mul_test():
    t1 = torch.tensor([[1, 2], [3, 4]])
    t2 = torch.tensor([[5, 6], [7, 8]])

    # 点乘：对应位置元素相乘，要求矩阵形状相同
    # * 或 mul()
    print(t1 * t2)
    print(t1.mul(t2))

    # Tensor乘法：必须能相乘，“中间相等取两头”
    # @ 或 matmul()
    print(t1 @ t2)
    print(t1.matmul(t2))

"""4. 更改tensor形状 reshape()和view()"""
def reshape_test():
    t1 = torch.tensor([[1, 2, 3], [4, 5, 6]])
    print(t1.is_contiguous())    # True
    t2 = t1.view(3, -1)
    print(t2.is_contiguous())    # True
    print(t2)

    t3 = t1.t()
    print(t3.is_contiguous())    # False
    # view()方法效率更高，一般内存不连续再选择reshape()
    # 且view() 返回的是视图

"""5. 交换张量的维度
transpose():交换tensor的两个维度（所以torch框架用的很少）
permute(): 任意排列tensor的维度
"""
def permute_test():
    t1 = torch.randint(0, 10, (2, 3, 4))
    print(t1)

    t2 = torch.permute(t1, (1, 2, 0)) # 现在应该是3*4*2
    print(t2)
    print(t2.shape)

"""6. 升维和降维
升维：unsqueeze()
降维：squeeze()
使用场景：一般在图像处理中，添加或删除维度
"""
def unsqueeze_test():
    t1 = torch.randint(0, 10, (2, 3, 4))
    print(t1.size())
    # 升维，法1
    t2 = torch.unsqueeze(t1,dim=0)
    print(t2.size())
    # 升维，法2
    t2 = t1.unsqueeze(0)
    print(t2.size())

    # 降维
    # 1.如果不指定dim，则默认删除所有维度数为1的维度
    # 2.如果指定dim的维度数不为1，则不做任何操作，也不报错
    t3 = torch.squeeze(t2, dim=1)
    print(t3.size())



if __name__ == '__main__':
    #get_element()
    #mul_test()
    #reshape_test()
    permute_test()