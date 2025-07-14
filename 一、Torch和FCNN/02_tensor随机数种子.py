import torch


def test_01():

    # 设置随机数种子，目的：使每次运行程序，产生的随机数相同
    torch.manual_seed(42)  # 每次随机出来的张量都一样了

    s = torch.randint(0, 10, (3, 4))
    print(s)

    # randn:符合标准正态分布的随机数
    s1 = torch.randn(3, 4)
    print(s1)

if __name__ == '__main__':
    test_01()