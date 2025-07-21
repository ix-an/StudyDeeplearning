import torch
import matplotlib.pyplot as plt

# 设置中文显示
plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False


def plot_activation(func, func_name, x_range=(-10, 10), n_points=100):
    """绘制激活函数及其导数"""
    # 生成输入数据
    x = torch.linspace(x_range[0], x_range[1], n_points)
    # 计算函数值
    y = func(x)

    # 计算导数（自动求导）
    x_grad = torch.linspace(x_range[0], x_range[1], n_points, requires_grad=True)
    y_grad = func(x_grad)
    y_grad.sum().backward()  # 标量反向传播
    grad = x_grad.grad.detach().numpy()

    # 绘图
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    # 函数曲线
    ax1.plot(x.numpy(), y.numpy())
    ax1.set_title(f"{func_name}函数")
    ax1.set_xlabel("x")
    ax1.set_ylabel("y")
    ax1.grid(True)
    # 导数曲线
    ax2.plot(x_grad.detach().numpy(), grad, color="red")
    ax2.set_title(f"{func_name}导数（📌考点：反向传播梯度计算）")
    ax2.set_xlabel("x")
    ax2.set_ylabel("梯度")
    ax2.grid(True)
    plt.show()


# 测试：绘制ReLU和Softmax（示例）
if __name__ == "__main__":
    # ReLU函数（📌考点：解决梯度消失，计算高效）
    plot_activation(torch.relu, "ReLU")


    # Softmax函数（输入为向量，这里简化为单值演示趋势）
    def softmax_demo(x):
        return torch.softmax(x, dim=0)  # 📌考点：dim参数需指定（通常为1，按行计算）


    plot_activation(softmax_demo, "Softmax", x_range=(-5, 5))