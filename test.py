import datetime
import logging
import os
import random
import shutil
import sys
import traceback
from argparse import ArgumentParser
from os.path import abspath, basename, dirname
from statistics import mean

import numpy as np
import setproctitle
import torch
import torch.nn as nn
import torch.nn.functional as F
from tensorboardX import SummaryWriter
from torch.optim import SGD, Adadelta, Adam
from torch.utils.data import DataLoader
from tqdm import tqdm

try:
    from nets import LossWrapper
except:
    path = dirname(dirname(abspath(__file__)))
    sys.path.append(path)
    from nets import LossWrapper
finally:
    from nets import IOUMetric, get_eccnet, GetMyDeepLab, get_CCNET_Model
    from utils import ListDataSet


input = torch.randn(2, 4, 256, 256).cuda()
net = get_CCNET_Model()
y = net(input)
print(y, y[0].shape)
