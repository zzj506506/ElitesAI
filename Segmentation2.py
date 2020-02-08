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
        self.resnet_layer = model

        self.conv = nn.Conv2d(in_channels=512,out_channels=num_classes, kernel_size=1)
        self.conv_t=nn.ConvTranspose2d(in_channels=num_classes,out_channels=num_classes, kernel_size=64, padding=16,stride=32)
        # self.sigmoid=nn.Sigmoid()

    def forward(self, x):
        x = self.resnet_layer(x)

        x = self.conv(x)
        x=self.conv_t(x)
        # return self.sigmoid(x)
        return x

net=Net(net,num_classes)



def bilinear_kernel(in_channels, out_channels, kernel_size):
   factor = (kernel_size + 1) // 2
   if kernel_size % 2 == 1:
       center = factor - 1
   else:
       center = factor - 0.5
   og = np.ogrid[:kernel_size, :kernel_size]
   filt = (1 - abs(og[0] - center) / factor) * (1 - abs(og[1] - center) / factor)
   weight = np.zeros((in_channels, out_channels, kernel_size, kernel_size), dtype='float32')
   weight[range(in_channels), range(out_channels), :, :] = filt
   return torch.from_numpy(weight)

conv_trans = nn.ConvTranspose2d(in_channels=3,out_channels=3, kernel_size=4, padding=1, stride=2)
conv_trans.weight=nn.Parameter(torch.Tensor(bilinear_kernel(3, 3, 4)))
img = image.open('./img/catdog.jpg')
X=np.array(img.getdata()).reshape(img.size[1], img.size[0], 3)
print(X.shape)
X = np.expand_dims(X.astype('float32').transpose((2, 0, 1)),axis=0) / 255
X=torch.Tensor(X)
Y = conv_trans(X)
out_img = Y[0].transpose(1, 0).transpose(0,2)
print(out_img.shape)
plt.figure(figsize=(10,8))
plt.subplot(2,1,1)
plt.imshow(img)
plt.axis('off')
plt.subplot(2,1,2)
plt.imshow(ToPILImage()(out_img))
plt.axis('off')
plt.show()

# nn.init.constant(net.conv_t.weight,torch.Tensor(bilinear_kernel(num_classes, num_classes, 64)))
nn.init.xavier_uniform(net.conv.weight)
net.conv_t.weight=nn.Parameter(torch.Tensor(bilinear_kernel(num_classes, num_classes, 64)))
print(net.conv.weight)
print(net.conv_t.weight)
net.cuda()
VOC_COLORMAP = [[0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0],
                [0, 0, 128], [128, 0, 128], [0, 128, 128], [128, 128, 128],
                [64, 0, 0], [192, 0, 0], [64, 128, 0], [192, 128, 0],
                [64, 0, 128], [192, 0, 128], [64, 128, 128], [192, 128, 128],
                [0, 64, 0], [128, 64, 0], [0, 192, 0], [128, 192, 0],
                [0, 64, 128]]
VOC_CLASSES = ['background', 'aeroplane', 'bicycle', 'bird', 'boat',
               'bottle', 'bus', 'car', 'cat', 'chair', 'cow',
               'diningtable', 'dog', 'horse', 'motorbike', 'person',
               'potted plant', 'sheep', 'sofa', 'train', 'tv/monitor']

colormap2label = np.zeros(256 ** 3)
for i, colormap in enumerate(VOC_COLORMAP):
    colormap2label[(colormap[0] * 256 + colormap[1]) * 256 + colormap[2]] = i


def voc_label_indices(colormap, colormap2label):
    idx = ((colormap[:, :, 0] * 256 + colormap[:, :, 1]) * 256+ colormap[:, :, 2])
    return colormap2label[idx]

def voc_rand_crop(data, label,img_w,img_h):
    data=image.fromarray(data.astype('uint8')).convert('RGB')
    # label=image.fromarray(label.astype('uint8')).convert('RGB')
    width1 = random.randint(0, data.size[1] - img_w)
    height1 = random.randint(0, data.size[0] - img_h)
    width2 = width1 + img_w
    height2 = height1 + img_h

    data = data.crop((width1, height1, width2, height2))
    label = label.crop((width1, height1, width2, height2))
    data=np.array(data.getdata()).reshape(data.size[1], data.size[0], 3)
    label = np.array(label.getdata()).reshape(label.size[1], label.size[0], 3)
    return data, label

voc_dir='./data/VOCtrainval_11-May-2012/VOCdevkit/VOC2012'
def read_voc_images(root=voc_dir, is_train=True):
    txt_fname = '%s/ImageSets/Segmentation/%s' % (
        root, 'train.txt' if is_train else 'val.txt')
    with open(txt_fname, 'r') as f:
        images = f.read().split()
    features, labels = [None] * len(images), [None] * len(images)
    for i, fname in enumerate(images):
        features[i] = image.open('%s/JPEGImages/%s.jpg' % (root, fname)).convert('RGB')
        labels[i] = image.open('%s/SegmentationClass/%s.png' % (root, fname)).convert('RGB')
    return features, labels

train_features, train_labels = read_voc_images()

class VOCSegDataset(Dataset):
    def __init__(self, is_train, crop_size, voc_dir, colormap2label):
        self.rgb_mean = np.array([0.485, 0.456, 0.406])
        self.rgb_std = np.array([0.229, 0.224, 0.225])
        self.crop_size = crop_size
        features, labels = read_voc_images(root=voc_dir, is_train=is_train)
        self.features = [self.normalize_image(feature)
                         for feature in self.filter(features)]
        self.labels = self.filter(labels)
        self.colormap2label = colormap2label
        print('read ' + str(len(self.features)) + ' examples')

    def normalize_image(self, img):
        img = np.array(img).reshape(img.size[1], img.size[0], 3)
        return (img.astype('float32') / 255 - self.rgb_mean) / self.rgb_std

    def filter(self, imgs):
        return [img for img in imgs if (
            img.size[1] > self.crop_size[0] and
            img.size[0] > self.crop_size[1])]

    def __getitem__(self, idx):
        feature, label = voc_rand_crop(self.features[idx], self.labels[idx],
                                       *self.crop_size)
        label=voc_label_indices(label, self.colormap2label)
        return (feature.transpose((2, 0, 1)),label)

    def __len__(self):
        return len(self.features)

crop_size=(320,480)
voc_train = VOCSegDataset(True, crop_size, voc_dir, colormap2label)
voc_test = VOCSegDataset(False, crop_size, voc_dir, colormap2label)

batch_size = 32
num_workers = 0 if sys.platform.startswith('win32') else 4
train_iter = DataLoader(voc_train, batch_size, shuffle=True, num_workers=num_workers)
test_iter = DataLoader(voc_test, batch_size,num_workers=num_workers)

# ctx = d2l.try_all_gpus()
# loss = gloss.SoftmaxCrossEntropyLoss(axis=1)
# net.collect_params().reset_ctx(ctx)
# trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': 0.1,
#                                                       'wd': 1e-3})
# d2l.train(train_iter, test_iter, net, loss, trainer, ctx, num_epochs=5)

def train(train_iter,test_iter,net,num_epochs=5):
    CE_loss=nn.CrossEntropyLoss()
    trainer=torch.optim.SGD(lr=0.1,weight_decay=1e-3,params=net.parameters())
    # net.cuda()
    # Tensor = torch.cuda.Tensor
    for epoch in range(1,num_epochs+1):
        for iter,(data,label) in enumerate(train_iter):
            trainer.zero_grad()
            loss=CE_loss(net(data.float().cuda()),label.long().cuda())
            loss.backward()
            trainer.step()
            print("epoch{}, iter{}, loss: {}".format(epoch, iter, loss.item()))


train(train_iter,test_iter,net)

def predict(img):
    X = img
    X = np.expand_dims(X,axis=0)
    X=torch.cuda.FloatTensor(X).transpose(1, 2).transpose(1, 3)
    pred = np.argmax(net(X).cpu().detach(), axis=1)
    return pred.reshape((pred.shape[1], pred.shape[2]))


def label2image(colormap,pred):
    X = pred.numpy().astype('int32').tolist()
    for i in range(len(X)):
        for j in range(len(X[0])):
            X[i][j]=colormap[X[i][j]]
    return np.array(X).astype('uint8')

test_images, test_labels = read_voc_images(is_train=False)
n, imgs = 4, []
for i in range(n):
    X=test_images[i].crop((0, 0, 480, 320))
    pred = label2image(VOC_COLORMAP,predict(X))
    imgs += [X, transforms.ToPILImage()(pred), test_labels[i].crop((0, 0, 480, 320))]

fig=plt.figure(figsize=(6,8))
for i in range(n):
    plt.subplot(n,3,i*3+1)
    plt.imshow(imgs[i*3])
    plt.axis('off')
    plt.subplot(n,3,i*3+2)
    plt.imshow(imgs[i*3+1])
    plt.axis('off')
    plt.subplot(n,3,i*3+3)
    plt.imshow(imgs[i*3+2])
    plt.axis('off')
plt.show()
# d2l.show_images(imgs[::3] + imgs[1::3] + imgs[2::3], 3, n);