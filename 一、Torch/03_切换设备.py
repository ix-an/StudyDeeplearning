import torch

def test01():
    # 法1：在创建张量时指定device
    t1 = torch.tensor([1,2,3], device="cuda")
    print(t1)

    # 法2：使用 to() 转换运算设备
    t2 = torch.tensor([1,2,3])
    print(t2.device)
    t2 = t2.to("cuda")
    print(t2.device)

    # 法3：使用 cpu() 和 cuda()
    t3 = torch.tensor([1,2,3], device="cuda")
    t3 = t3.cpu()
    print(t3.device)


if __name__ == "__main__":
    test01()