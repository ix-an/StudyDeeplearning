import torch

# Pytorch安装：Anaconda创建虚拟环境
print(torch.cuda.is_available())    # True
print(torch.__version__)    # 1.13.1+cu117
print(torch.version.cuda)    # 11.7

# 📌 CUDA版本不匹配会导致什么问题？
# 无法启用GPU加速，可能报"CUDA driver version is insufficient for CUDA runtime version"错误
# 需保证显卡驱动版本 ≥ CUDA版本。