import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset

"""自定义数据集类步骤：
1. 继承Dataset类
2. 实现__init__方法，初始化数据集
3. 实现__len__方法，返回数据集长度
4. 实现__getitem__方法，根据索引获取对应位置的数据
"""
class MyDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        sample = self.data[index]
        label = self.labels[index]

        return sample, label    # 返回的是元组

def test01():
    x = torch.randn(100, 20)
    y = torch.randn(100, 1)

    dataset = MyDataset(x, y)
    print(dataset[0])

    """DataLoader:数据加载器，返回迭代器，用来分批次加载数据
    参数：
        1. dataset：目标数据集
        2. batch_size：批次大小
        3. shuffle: 是否打乱数据
        4. num_workers: 加载数据的线程数
    """
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=10,
        shuffle=True,
        num_workers=0,
    )

    for sample, label in dataloader:
        print(sample)
        print(label)
        break

"""TensorDataset:pytorch提供的一个Dataset类
1. 可以将多个张量封装成一个Dataset类（*tensors）
2. 可以使用DataLoader加载数据
可以优先选择TensorDataset，因为它的封装性更好，使用起来更方便
有API优先用API，别人的API经过测试，更加稳定
"""
def test02():
    torch.manual_seed(23)
    x = torch.randn(100, 20)
    y = torch.randn(100, 1)

    dataset = TensorDataset(x, y)

    dataloader = DataLoader(
        dataset=dataset,
        batch_size=10,
        shuffle=True,
    )

    for sample, label in dataloader:
        print(sample)
        print(label)
        break


if __name__ == '__main__':
    # test01()
    test02()