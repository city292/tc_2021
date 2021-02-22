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
import cv2
import numpy as np
import setproctitle
import torch
import torch.nn as nn
import torch.nn.functional as F
from tensorboardX import SummaryWriter
from torch.optim import SGD, Adadelta, Adam
from torch.utils.data import DataLoader
from tqdm import tqdm
import zipfile

try:
    from nets import LossWrapper
except:
    path = dirname(dirname(abspath(__file__)))
    sys.path.append(path)
    from nets import LossWrapper
finally:
    from nets import IOUMetric, get_eccnet, get_attu_net, get_enet, GetMyDeepLabv3Plus, GetMyDeepLab, get_CCNET_Model
    from utils import ListDataSet, readlabel, readTiff


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


def net_pred(NET=None):

    def wapper(x):
        pred = NET(x)
        if isinstance(pred, list):
            pred = pred[0]
        return pred
    return wapper


def ttabatch2images(img, f, fold=8):
    img0 = img
    res0 = F.softmax(f(img0), dim=1)
    img1 = torch.rot90(img0, 1, [2, 3])
    res1 = torch.rot90(F.softmax(f(img1), dim=1), -1, [2, 3])

    img2 = torch.flip(img0, [2, ])
    res2 = torch.flip(F.softmax(f(img2), dim=1), [2, ])
    img3 = torch.flip(img1, [2, ])
    res3 = torch.rot90(torch.flip(F.softmax(f(img3), dim=1), [2, ]), -1, [2, 3])
    if fold == 4:
        return (res0 + res1 + res2 + res3) / 4
    img4 = torch.flip(img0, [3, ])
    res4 = torch.flip(F.softmax(f(img4), dim=1), [3, ])
    img5 = torch.flip(img1, [3, ])
    res5 = torch.rot90(torch.flip(F.softmax(f(img5), dim=1), [3, ]), -1, [2, 3])

    img6 = torch.flip(img2, [3, ])
    res6 = torch.flip(torch.flip(F.softmax(f(img6), dim=1), [3, ]), [2, ])
    img7 = torch.flip(img3, [3, ])
    res7 = torch.rot90(torch.flip(torch.flip(F.softmax(f(img7), dim=1), [3, ]), [2, ]), -1, [2, 3])

    return (res0 + res1 + res2 + res3 + res4 + res5 + res6 + res7) / 8


torch.backends.cudnn.enabled = True
np.set_printoptions(precision=2)

npgus = 1
get_gpuid(npgus)
filelst = []
train_dir = '/media/l/e6aa5997-4a1e-42e4-8782-83e2693751bd/city/data/tc_2021/suichang_round1_test_partA_210120/'
res_dir = '/media/l/e6aa5997-4a1e-42e4-8782-83e2693751bd/city/tmp/result/'
if os.path.exists(res_dir):
    shutil.rmtree(res_dir)
os.mkdir(res_dir)
# ccnet 0.3463
# deeplabv3 0.336
# deeplabv3plus 0.3375
# eccnet 0.3378
# attu_net 0.3

ckpt = torch.load('/media/l/e6aa5997-4a1e-42e4-8782-83e2693751bd/city/logs/CITY_tc_attu_net_21-02-09_0/E_85.ckpt')
setproctitle.setproctitle('CITY_' + ckpt['args'].segname + '_eval')
if ckpt['args'].segname == 'attu_net':
    NET = get_attu_net(gpu_ids=1, num_classes=10)
elif ckpt['args'].segname == 'eccnet':
    NET = get_eccnet(gpu_ids=1, num_classes=10)
elif ckpt['args'].segname == 'enet':
    NET = get_enet(gpu_ids=1, num_classes=10)
elif ckpt['args'].segname == 'deeplabv3plus':
    NET = GetMyDeepLabv3Plus(gpu_ids=1, num_classes=10)
elif ckpt['args'].segname == 'deeplabv3':
    NET = GetMyDeepLab(gpu_ids=1, num_classes=10)
elif ckpt['args'].segname == 'ccnet':
    NET = get_CCNET_Model(gpu_ids=1, num_classes=10)

NET.load_state_dict(ckpt['segnet'])
zip_name = '/media/l/e6aa5997-4a1e-42e4-8782-83e2693751bd/city/tmp/result_' + ckpt['args'].segname + '.zip'
NET.eval()
with torch.no_grad():
    if os.path.exists(zip_name):
        os.remove(zip_name)
    with zipfile.ZipFile(zip_name, mode='w') as zipf:

        for name in tqdm(os.listdir(train_dir), disable=False):
            if not name.endswith('.tif'):
                continue
            image = readTiff(os.path.join(train_dir, name))
            image = torch.from_numpy(image).float().div(255)
            image = torch.unsqueeze(image, 0).cuda()
            # pred = NET(image)
            # if isinstance(pred, list):
            #     pred = pred[0]

            pred = ttabatch2images(image, net_pred(NET), fold=8)

            output_img = torch.squeeze(torch.argmax(pred, dim=1)).cpu().numpy() + 1

            cv2.imwrite(os.path.join(res_dir, name[:-4] + '.png'), output_img)
            zipf.write(os.path.join(res_dir, name[:-4] + '.png'), name[:-4] + '.png')
