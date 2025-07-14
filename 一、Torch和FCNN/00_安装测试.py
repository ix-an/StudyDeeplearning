import torch

# Pytorch安装：Anaconda创建虚拟环境
print(torch.cuda.is_available())    # True
print(torch.__version__)    # 1.13.1+cu117