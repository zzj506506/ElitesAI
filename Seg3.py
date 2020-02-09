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

def voc_rand_crop(data, label, height, width):
    data, rect = transforms.RandomCrop((height, width))(data)
    label = transforms.FixedCrop(*rect)(label)
    return data, label

train_features, train_labels = read_voc_images()
n = 5
imgs = []
for _ in range(n):
    imgs += voc_rand_crop(train_features[0], train_labels[0], 200, 300)
fig=plt.figure(figsize=(10,4))
for i in range(5):
    plt.subplot(2,5,i+1)
    plt.imshow(imgs[2*i])
    plt.axis('off')
for i in range(5):
    plt.subplot(2,5,i+6)
    plt.imshow(imgs[2*i+1])
    plt.axis('off')
plt.show()

