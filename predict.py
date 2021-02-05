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

try:
    from nets import LossWrapper
except:
    path = dirname(dirname(abspath(__file__)))
    sys.path.append(path)
    from nets import LossWrapper
finally:
    from nets import IOUMetric, get_eccnet
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

# To save the checkpoint


def save_checkpoint(state, save_path):
    try:
        torch.save(state, save_path, _use_new_zipfile_serialization=False)
    except Exception as e:
        torch.save(state, save_path)


def ttabatch2images(img, f, fold=8):
    sigmoid = torch.nn.Sigmoid()
    img0 = img
    res0 = sigmoid(f(img0))
    img1 = torch.rot90(img0, 1, [2, 3])
    res1 = torch.rot90(sigmoid(f(img1)), -1, [2, 3])

    img2 = torch.flip(img0, [2, ])
    res2 = torch.flip(sigmoid(f(img2)), [2, ])
    img3 = torch.flip(img1, [2, ])
    res3 = torch.rot90(torch.flip(sigmoid(f(img3)), [2, ]), -1, [2, 3])
    if fold == 4:
        return (res0 + res1 + res2 + res3) / 4
    img4 = torch.flip(img0, [3, ])
    res4 = torch.flip(sigmoid(f(img4)), [3, ])
    img5 = torch.flip(img1, [3, ])
    res5 = torch.rot90(torch.flip(sigmoid(f(img5)), [3, ]), -1, [2, 3])

    img6 = torch.flip(img2, [3, ])
    res6 = torch.flip(torch.flip(sigmoid(f(img6)), [3, ]), [2, ])
    img7 = torch.flip(img3, [3, ])
    res7 = torch.rot90(torch.flip(torch.flip(sigmoid(f(img7)), [3, ]), [2, ]), -1, [2, 3])

    return (res0 + res1 + res2 + res3 + res4 + res5 + res6 + res7) / 8

torch.backends.cudnn.enabled = True
np.set_printoptions(precision=2)

setproctitle.setproctitle('CITY_ECC_eval')
npgus = 1
get_gpuid(npgus)
filelst = []
train_dir = '/media/l/e6aa5997-4a1e-42e4-8782-83e2693751bd/city/data/tc_2021/suichang_round1_test_partA_210120/'
res_dir = '/media/l/e6aa5997-4a1e-42e4-8782-83e2693751bd/city/tmp/tc_res/'
if os.path.exists(res_dir):
    shutil.rmtree(res_dir)
os.mkdir(res_dir)
NET = get_eccnet(gpu_ids=npgus, num_classes=10)
ckpt = torch.load('/media/l/e6aa5997-4a1e-42e4-8782-83e2693751bd/city/logs/CITY_ECC_21-02-04_6/E_65.ckpt')
NET.load_state_dict(ckpt)

NET.eval()
with torch.no_grad():
    for name in tqdm(os.listdir(train_dir), disable=True):
        if not name.endswith('.tif'):
            continue
        image = readTiff(os.path.join(train_dir, name))
        image = torch.from_numpy(image).float().div(255)
        image = torch.unsqueeze(image, 0).cuda()
        pred = NET(image)
        if isinstance(pred, list):
            pred = pred[0]

        output_img = torch.squeeze(torch.argmax(F.softmax(pred, dim=1), dim=1)).cpu().numpy() + 1

        cv2.imwrite(os.path.join(res_dir, name[:-4] + '.png'), output_img)
