import torch

"""自动微分（求导）：
1. 叶子结点张量要添加 requires_grad=True
2. 叶子结点的数据类型必须为 float
3. 调用 backward() 方法进行反向传播，自动求导
4. 求导后的梯度会保存在叶子结点的 grad 属性中

"""
def test01():
    """标量梯度计算"""
    x = torch.tensor(2.33, requires_grad=True)

    y = x.pow(2)

    # 反向传播，做自动求导
    y.backward()
    print(x.grad)

def test02():
    """向量梯度计算"""
    x = torch.tensor([1,2,3], requires_grad=True, dtype=torch.float32)

    y = x.pow(2)    # [1, 4, 9]

    # 不能直接 y.backward()，因为y是一个向量，无法计算梯度
    # 会报错：RuntimeError: grad can be implicitly created only for scalar outputs
    # 意思是：只能对标量进行自动求导

    # 解决办法1：创建梯度张量，形状和y相同
    #y.backward(torch.tensor([1.0,1.0,1.0]))
    # 不推荐，因为你创建的梯度向量必须和y的形状相同

    # 解决办法2：将y.sum()求和，求和之后再进行反向传播（常用）
    z = y.sum()    # z = y1 + y2 + y3 ，求导后是3个1，对最终结果无影响
    z.backward()
    """把梯度向量通过类似计算损失的方式将输出转换为标量，然后再调用backward()"""
    print(x.grad)

def test03():
    """多标量梯度计算"""
    x = torch.tensor([1,2,3], requires_grad=True, dtype=torch.float32)
    y = torch.tensor([2,3,4], requires_grad=True, dtype=torch.float32)

    z = x.mul(y)    # [2, 6, 12]
    print(z.requires_grad)    # True

    loss = z.sum()

    loss.backward()

    print(x.grad, y.grad)
    # print(z.grad)    # None，且会警告
    """中间变量在反向传播中，也会参与梯度计算，但是计算完成后，将被释放"""

def test04():
    """控制梯度计算"""
    x = torch.tensor([1,2,3], requires_grad=True, dtype=torch.float32)
    y = x.pow(2)
    print(y.requires_grad)    # True

    # torch.no_grad() 禁止该上下文的代码参与梯度计算
    with torch.no_grad():
        z = y.pow(2)
        print(z.requires_grad)    # False


def test05():
    """叶子结点的梯度自动累加 -> 每轮backward前梯度清零"""
    x = torch.tensor([1,2,3], requires_grad=True, dtype=torch.float32)

    # x和y的映射函数要放在循环内部，否则会错误释放
    # 计算图中叶子结点的梯度默认是累加的
    # 不希望叶子结点的梯度累加，需要对每轮次的梯度进行清零
    for epoch in range(3):
        y = x.pow(2)
        z = y.sum()

        # 梯度清零：一般在backward()之前进行
        if x.grad is not None:
            x.grad.zero_()
        z.backward()
        print(x.grad)

if __name__ == '__main__':
    #test01()
    #test02()
    #test03()
    #test04()
    test05()