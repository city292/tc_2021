import codecs
import random

import numpy as np
import torch
import torchvision
from PIL import Image
from cv2 import imread
from torch.utils.data import Dataset


def readTiff(filename):

    return np.transpose(np.array(Image.open(filename), np.int), [2, 0, 1])


def readlabel(labelname):
    y = Image.open(labelname).convert('L')
    y = np.array(y, np.int)
    return y


class ListDataSet(Dataset):
    def __init__(self, files, label_smooth=False):

        self.files = files
        self.labelSmooth = label_smooth
        self.max_len = len(self.files)

    def __len__(self):
        return self.max_len

    def __getitem__(self, index):
        image = readTiff(self.files[index][0])
        # image = image + np.random.randint(-5, 5, image.shape)
        image = np.clip(image, 0, 255)
        xsize = image.size
        label = readlabel(self.files[index][1])
        ysize = label.size
        image = torch.from_numpy(image).float().div(255)

        if random.randint(0, 1) > 0:
            image = torch.flip(image, [1, ])
            # label = np.flip(label, 0)
            label = label[::-1, :].copy()
        if random.randint(0, 1) > 0:
            image = torch.flip(image, [2, ])
            label = label[:, ::-1].copy()
        # if random.randint(0, 1) > 0:
        #     image = torch.rot90(image, 1, [1, 2]))
        #     label = np.rot90(label)
        label = label - 1

        return image, label


class ListDataSetB3(Dataset):
    def __init__(self, files, label_smooth=False):

        self.files = files
        self.labelSmooth = label_smooth
        self.max_len = len(self.files)

    def __len__(self):
        return self.max_len

    def __getitem__(self, index):
        image = np.transpose(np.array(Image.open(self.files[index][0]).convert('RGB'), np.int), [2, 0, 1])
        # image = image + np.random.randint(-5, 5, image.shape)
        # image = np.clip(image, 0, 255)
        xsize = image.size
        label = readlabel(self.files[index][1])
        ysize = label.size
        image = torch.from_numpy(image).float().div(255)

        if random.randint(0, 1) > 0:
            image = torch.flip(image, [1, ])
            # label = np.flip(label, 0)
            label = label[::-1, :].copy()
        if random.randint(0, 1) > 0:
            image = torch.flip(image, [2, ])
            label = label[:, ::-1].copy()
        # if random.randint(0, 1) > 0:
        #     image = torch.rot90(image, 1, [1, 2]))
        #     label = np.rot90(label)
        label = label - 1

        return image, label
