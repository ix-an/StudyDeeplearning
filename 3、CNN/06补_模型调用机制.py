import torch
import torch.nn as nn

class TestModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(2, 1)
    def forward(self, x):
        return self.fc(x)

model = TestModel()
x = torch.tensor([[1.0, 2.0]], requires_grad=True)

# 正确调用：触发梯度计算
output1 = model(x)
output1.sum().backward()
print("正确调用的梯度：", model.fc.weight.grad)  # 有梯度值

# 重置梯度
model.zero_grad()

# 错误调用：梯度可能为0
output2 = model.forward(x)
output2.sum().backward()
print("直接调用forward的梯度：", model.fc.weight.grad)  # 可能为0或错误值