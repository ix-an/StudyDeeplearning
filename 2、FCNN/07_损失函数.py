import torch
from torch import nn

def test01():
    logits = torch.tensor([[1.5, 2.0, 0.5],[0.5, 1.0, 1.5]])

    labels = torch.tensor([1, 2])

    criterion = nn.CrossEntropyLoss()

    loss = criterion(logits, labels)

    print(loss.item())

def test02():

    # y 是模型的输出，已经被sigmoid处理过，确保其值域在(0,1)
    y = torch.tensor([[0.7], [0.2], [0.9], [0.7]])
    # targets 是真实的标签，0或1
    t = torch.tensor([[1], [0], [1], [0]], dtype=torch.float)

    criterion = nn.BCELoss()

    loss = criterion(y, t)

    print(loss.item())


if __name__ == '__main__':
    # test01()
    test02()