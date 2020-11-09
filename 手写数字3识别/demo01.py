import json
import torch
from PIL import Image
import torch.nn as nn
import torchvision
import matplotlib.pyplot as plt
from torchvision import transforms
import numpy as np
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import cv2 as cv



class ConvNet(nn.Module):
    def __init__(self, num_classes=10):
        super(ConvNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.fc = nn.Linear(7 * 7 * 32, num_classes)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        return out


def test_mydata():

    im = plt.imread('new0.jpg')       # 读入图片，需要黑底白字

    images = Image.open('new0.jpg')    # 将图片存储到images里面
    images = images.resize((28, 28))   # 图片大小调整为28*28
    images = images.convert('L')       # 灰度化
    transform = transforms.ToTensor()
    images = transform(images)
    images = images.resize(1, 1, 28, 28)

    # 加载网络和参数
    model = ConvNet()
    model.load_state_dict(torch.load('model.ckpt'))
    model.eval()
    outputs = model(images)
    values, indices = outputs.data.max(1)  # 返回最大概率值和下标
    # plt.title('{}'.format(int(indices[0])))
    if indices[0]==3:
        plt.title('this is 3')
    else:
        plt.title('this is not 3')

    plt.imshow(im)
    plt.show()


test_mydata()


# def test_MNISTdata():
#     test_set = torchvision.datasets.MNIST(
#         root='data/'  # 数据文件位置
#         , train=False
#         , download=False
#         , transform=transforms.Compose([
#             transforms.ToTensor()
#         ])
#     )
#     test_loader = torch.utils.data.DataLoader(
#         test_set, batch_size=10
#     )
#     batch = next(iter(test_loader))
#     # 加载网络和参数
#     images, labels = batch
#     model = ConvNet()
#     model.load_state_dict(torch.load('model.ckpt'))
#     model.eval()
#     outputs = model(images)
#     grid = torchvision.utils.make_grid(images, nrow=10)  # make_grid的作用是将若干幅图像拼成一幅图像。
#     plt.imshow(np.transpose(grid, (1, 2, 0)))  # 转置，调整图片显示
#     values, indices = outputs.data.max(1)
#     plt.title('{}'.format(indices))
#     plt.show()

