import torch
from cv2.typing import map_int_and_double
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
def test01():
    path = '../datasets/images/100.jpg'

    img = Image.open(path)
    # print(img.size, img)

    transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor()
    ])

    t_img = transform(img)
    print(t_img.size(), t_img)

    t_img = torch.permute(t_img, (1, 2, 0))

    plt.imshow(t_img)
    plt.show()

def test02():
    path = '../datasets/images/100.jpg'

    img = Image.open(path)
    # print(img.size, img)

    transform = transforms.Compose([
        # 随机裁剪
        transforms.RandomCrop(size=(224, 224)),
        transforms.ToTensor()
    ])

    t_img = transform(img)
    print(t_img.size(), t_img)

    t_img = torch.permute(t_img, (1, 2, 0))

    plt.imshow(t_img)
    plt.show()

def test03():
    path = '../datasets/images/100.jpg'

    img = Image.open(path)

    transform = transforms.Compose([
        # 随机水平翻转
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor()
    ])

    t_img = transform(img)
    print(t_img.size(), t_img)

    t_img = torch.permute(t_img, (1, 2, 0))

    plt.imshow(t_img)
    plt.show()

def test04():
    path = '../datasets/images/100.jpg'

    img = Image.open(path)
    # print(img.size, img)

    transform = transforms.Compose([
        # 随机旋转
        # degrees参数：degrees=30，表示在(-30, 30)之间随机旋转
        # degree=(30,60)表示在这个范围内随机旋转
        transforms.RandomRotation((30, 90)),
        transforms.ToTensor()
    ])

    t_img = transform(img)
    print(t_img.size(), t_img)

    t_img = torch.permute(t_img, (1, 2, 0))

    plt.imshow(t_img)
    plt.show()


def test05():
    t = torch.randn(3, 224, 224)
    transform = transforms.Compose([
        # 张量转图片
        transforms.ToPILImage()
    ])

    img = transform(t)
    print(img.size)

    img.show()


if __name__ == '__main__':
    # test01()
    # test02()
    # test03()
    # test04()
    test05()