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
    from nets import IOUMetric, get_eccnet
    from utils import ListDataSet


def get_gpuid(ngpus=1):
    tmpfile = 'tmp_gpu_id'
    tmppre = '/media/l/e6aa5997-4a1e-42e4-8782-83e2693751bd/city/tmp'

    if os.path.exists(tmppre):
        tmpfile = os.path.join(tmppre, tmpfile)
    if os.path.exists(tmpfile):
        os.remove(tmpfile)
    os.system('nvidia-smi -q -d Memory |grep -A4 GPU|grep Free > ' + tmpfile)
    memory_gpu = [int(x.split()[2]) for x in open(tmpfile, 'r').readlines()]
    if os.path.exists(tmpfile):
        os.remove(tmpfile)
    memory_gpu = sorted(list(zip(memory_gpu, list(range(len(memory_gpu))))),
                        key=lambda x: x[0])
    memory_gpu.reverse()
    tmp = ''
    for mem, id in memory_gpu:
        tmp += 'GPU%d: %d ' % (id, mem)
    logging.info(tmp)
    res = ''
    for i in range(ngpus):
        if len(res) > 0:
            res += ','
        res += str(memory_gpu[i][1])
    os.environ["CUDA_VISIBLE_DEVICES"] = res
    logging.info("CUDA_VISIBLE_DEVICES: " + res)
    return res

# To save the checkpoint


def save_checkpoint(state, save_path):
    try:
        torch.save(state, save_path, _use_new_zipfile_serialization=False)
    except Exception as e:
        torch.save(state, save_path)


torch.backends.cudnn.enabled = True
np.set_printoptions(precision=2)

setproctitle.setproctitle('CITY_ECC_eval')
npgus = 3
get_gpuid(npgus)
filelst = []
train_dir = '/media/l/e6aa5997-4a1e-42e4-8782-83e2693751bd/city/data/tc_2021/suichang_round1_train_210120/'
for name in os.listdir(train_dir):
    if name.endswith('.tif'):
        if os.path.exists(os.path.join(train_dir, name[:-4] + '.png')):
            filelst.append((os.path.join(train_dir, name), os.path.join(train_dir, name[:-4] + '.png')))
train_a_dataset = ListDataSet(filelst)

NET = get_eccnet(gpu_ids=npgus, num_classes=10)
ckpt = torch.load('/media/l/e6aa5997-4a1e-42e4-8782-83e2693751bd/city/logs/CITY_ECC_21-02-04_0/E_133_0.29.ckpt')
NET.load_state_dict(ckpt)
train_loader = DataLoader(train_a_dataset, batch_size=32 * npgus, num_workers=4, pin_memory=True)

with torch.no_grad():

    iou = IOUMetric(10)

    for img, label in tqdm(train_loader, ncols=80, disable=True):
        pred = NET(img)

        iou.add_batch(torch.argmax(F.softmax(pred[0], dim=1), dim=1).cpu().numpy(), label.numpy())

    print(iou.evaluate())
# {
#     1: "耕地",
#     2: "林地",
#     3: "草地",
#     4: "道路",
#     5: "城镇建设用地",
#     6: "农村建设用地",
#     7: "工业用地",
#     8: "构筑物"
#     9: "水域"
#     10: "裸地"
# }
