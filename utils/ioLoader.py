import codecs
import random

import numpy as np
import torch
import torchvision
from PIL import Image
from cv2 import imread
from torch.utils.data import Dataset
from osgeo import ogr, osr, gdal


def readTiff(filename):
    tif1 = gdal.Open(filename, gdal.GA_Update)
    tif_num_1 = tif1.ReadAsArray()
    return tif_num_1


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
        xsize = image.size
        label = readlabel(self.files[index][1])
        ysize = label.size
        image = torch.from_numpy(image).float().div(255)

        if xsize == (1024, 1024) and image.shape[-1] == 1024:
            x0 = random.randint(0, 1024 - 256)
            y0 = random.randint(0, 1024 - 256)
            image = image[:, x0:x0 + 256, y0:y0 + 256]
            if ysize == (1024, 1024):
                label = label[x0:x0 + 256, y0:y0 + 256]
        if random.randint(0, 1) > 0:
            image = torch.flip(image, [1, ])
            # label = np.flip(label, 0)
            label = label[::-1, :].copy()
        if random.randint(0, 1) > 0:
            image = torch.flip(image, [2, ])
            label = label[:, ::-1].copy()

            # label = np.flip(label, 1)
        label = label - 1
        # label = np.eye(10)[label - 1]
        if self.labelSmooth:
            label = label * 0.8 + 0.1

        return image, label
