import torch
from torch import nn

def test01():
    x = torch.randint(0, 10, (5, 6), dtype=torch.float32)

    dropout = nn.Dropout(p=0.5)

    print(x)    # 丢弃之前

    print(dropout(x))    # 丢弃之后
    """
    dropout之后：
        1. 每个元素有p的概率被丢弃
        2. 丢弃元素被置为0
        3. 保留元素：x * 1/(1-p)
    """


if __name__ == '__main__':
    test01()