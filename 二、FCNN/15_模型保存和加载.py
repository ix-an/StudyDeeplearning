import torch
import torch.nn as nn

class MyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 64)
        self.fc2 = nn.Linear(64, 5)

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        return x

"""保存模型"""
def save_model():
    model = MyNet()
    print(model)
    torch.save(model, './model/study_try.pt')

"""加载模型"""
def load_model():
    model = torch.load('./model/study_try.pt')
    print(model)

"""保存模型参数
如果保存的是模型参数，加载的是字典，内容是模型参数，并不是完整的模型
需要实现初始化模型，然后把模型参数导入到模型中
"""
def save_model_params():
    model = MyNet()
    state_dict = model.state_dict()
    torch.save(state_dict, './model/study_try_params.pt')

"""加载模型参数"""
def load_model_params():
    model = MyNet()
    state_dict = torch.load('./model/study_try_params.pt')
    model.load_state_dict(state_dict)
    print(model)


if __name__ == '__main__':
    # save_model()
    # load_model()
    # save_model_params()
    load_model_params()
