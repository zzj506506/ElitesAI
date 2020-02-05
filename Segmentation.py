import random

import matplotlib.pyplot as plt
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
n = 5
# imgs = train_features[0:n] + train_labels[0:n]
# fig=plt.figure(figsize=(10,4))
# for i in range(10):
#     plt.subplot(2,5,i+1)
#     plt.imshow(imgs[i])
#     plt.axis('off')
# plt.show()

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


# img=np.array(train_labels[0].getdata()).reshape(train_labels[0].size[1],train_labels[0].size[0],3)
# y = voc_label_indices(img, colormap2label)
# print(y[105:115, 130:140])
# print(VOC_CLASSES[1])


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



# imgs = []
# for _ in range(n):
#     imgs += voc_rand_crop(np.array(train_features[0].getdata()).reshape(train_features[0].size[1],train_features[0].size[0],3), train_labels[0], 200, 300)
# fig=plt.figure(figsize=(10,4))
# for i in range(5):
#     plt.subplot(2,5,i+1)
#     plt.imshow(imgs[2*i])
#     plt.axis('off')
# for i in range(5):
#     plt.subplot(2,5,i+6)
#     plt.imshow(imgs[2*i+1])
#     plt.axis('off')
# plt.show()

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
        return (feature.transpose((2, 0, 1)),
                voc_label_indices(label, self.colormap2label))

    def __len__(self):
        return len(self.features)

crop_size=(320,480)
voc_train = VOCSegDataset(True, crop_size, voc_dir, colormap2label)
voc_test = VOCSegDataset(False, crop_size, voc_dir, colormap2label)

batch_size = 64
num_workers = 0 if sys.platform.startswith('win32') else 4
train_iter = DataLoader(voc_train, batch_size, shuffle=True, num_workers=num_workers)
test_iter = DataLoader(voc_test, batch_size,num_workers=num_workers)


for X, Y in train_iter:
    print(X.shape)
    print(Y.shape)
    break