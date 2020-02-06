import random

import matplotlib.pyplot as plt
from torchvision.transforms import ToPILImage
# from matplotlib import image
from PIL import Image as image
from torch.utils.data import DataLoader,Dataset
from torch import nn
import numpy as np
from torch.autograd import Variable
import torch
from torchvision.datasets import ImageFolder
from torchvision.transforms import transforms
import zipfile
import os
from enum import Enum
import sys
from scipy import misc
from torchvision.models import resnet18
cuda = True if torch.cuda.is_available() else False

X = torch.Tensor(np.arange(1, 17).reshape((1, 1, 4, 4)))
K = torch.Tensor(np.arange(1, 10).reshape((1, 1, 3, 3)))
conv = nn.Conv2d(in_channels=1,out_channels=1, kernel_size=3)
conv.weight=nn.Parameter(K)
print(conv(X))
print(conv.weight)

W, k = np.zeros((4, 16)), np.zeros(11)
k[:3], k[4:7], k[8:] = K[0, 0, 0, :], K[0, 0, 1, :], K[0, 0, 2, :]
W[0, 0:11], W[1, 1:12], W[2, 4:15], W[3, 5:16] = k, k, k, k
print(np.dot(W, X.reshape(16)).reshape((1, 1, 2, 2)))
print(W)

conv = nn.Conv2d(in_channels=3,out_channels=10, kernel_size=4, padding=1, stride=2)
X = torch.Tensor(np.random.uniform(size=(1, 3, 64, 64)))
Y = conv(X)
print(Y.shape)

conv_trans = nn.ConvTranspose2d(in_channels=10,out_channels=3, kernel_size=4, padding=1, stride=2)
print(conv_trans(Y).shape)

pretrained_net = resnet18(pretrained=True)
print(pretrained_net)

net=nn.Sequential(*list(pretrained_net.children())[:-2])
X = torch.Tensor(np.random.uniform(size=(1, 3, 320, 480)))
print(net(X).shape)

num_classes = 21


class Net(nn.Module):
    def __init__(self, model,num_classes):
        super(Net, self).__init__()
        # 取掉model的后两层
        self.resnet_layer = model

        self.conv = nn.Conv2d(in_channels=512,out_channels=num_classes, kernel_size=1)
        self.conv_t=nn.ConvTranspose2d(in_channels=num_classes,out_channels=num_classes, kernel_size=64, padding=16,
                           stride=32)

    def forward(self, x):
        x = self.resnet_layer(x)

        x = self.conv(x)
        x=self.conv_t(x)
        return x

net=Net(net,num_classes)


def bilinear_kernel(in_channels, out_channels, kernel_size):
    factor = (kernel_size + 1) // 2
    if kernel_size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = np.ogrid[:kernel_size, :kernel_size]
    filt = (1 - abs(og[0] - center) / factor) * \
           (1 - abs(og[1] - center) / factor)
    weight = np.zeros((in_channels, out_channels, kernel_size, kernel_size),
                      dtype='float32')
    weight[range(in_channels), range(out_channels), :, :] = filt
    return np.array(weight)

conv_trans = nn.ConvTranspose2d(in_channels=3,out_channels=3, kernel_size=4, padding=1, stride=2)
conv_trans.weight=nn.Parameter(torch.Tensor(bilinear_kernel(3, 3, 4)))
img = image.open('../img/catdog.jpg')
img=np.array(img.getdata()).reshape(img.size[1], img.size[0], 3)
X = img.astype('float32').transpose((2, 0, 1)).expand_dims(axis=0) / 255
X=torch.Tensor(X)
Y = conv_trans(X)
out_img = Y[0].transpose((1, 2, 0))
print(img.shape)
print(out_img.shape)
plt.figure(figsize=(10,8))
plt.imshow(image.fromarray(img.astype('uint8')).convert('RGB'))
plt.imshow(ToPILImage()(out_img))