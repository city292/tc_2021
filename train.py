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
from torch.utils.data.dataset import random_split
from tensorboardX import SummaryWriter
from torch.optim import SGD, Adadelta, Adam
from torch.utils.data import DataLoader
from tqdm import tqdm
from time import time
try:
    from nets import LossWrapper
except:
    path = dirname(dirname(abspath(__file__)))
    sys.path.append(path)
    from nets import LossWrapper
finally:
    from nets import IOUMetric, get_eccnet
    from utils import ListDataSet


def beijing(sec, what):
    beijing_time = datetime.datetime.now() + datetime.timedelta(hours=8)
    return beijing_time.timetuple()


def set_logger(logfile, on_NCloud=False):
    if on_NCloud:
        logging.Formatter.converter = beijing
    logger = logging.getLogger()  # 不加名称设置root logger
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
        '%(asctime)s %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S')

    # 使用FileHandler输出到文件
    fh = logging.FileHandler(logfile, mode='w')
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)

    # 使用StreamHandler输出到屏幕
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)

    # 添加两个Handler
    logger.addHandler(ch)
    logger.addHandler(fh)


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


def get_dir(filedir=None, nb_params=0, str=''):
    now = datetime.datetime.now()
    day = now.strftime('%y-%m-%d')

    for i in range(1000):
        tem = '%s_%s_%d' % (str, day, i)
        if os.path.exists(os.path.join(filedir, tem)):
            continue
        os.mkdir(os.path.join(filedir, tem))

        return os.path.join(filedir, tem)
    return None


class Criterion(nn.Module):
    def __init__(self, cri1, cri2=None):
        super(Criterion, self).__init__()
        self.cri1 = cri1
        self.cri2 = cri2

    def forward(self, preds, labels):
        if self.cri2 is None:
            return self.cri1(preds, labels)
        else:
            return self.cri1(preds, labels) * 0.5 + self.cri2(preds, labels) * 0.5


def get_args():
    parser = ArgumentParser(description='CITY_ECC SEGMENTATION PyTorch')
    parser.add_argument('--ngpus', type=int, default=3, help='GPU数')
    parser.add_argument('--epochs', type=int, default=1000, help='最大迭代epoch数')
    parser.add_argument('--use_tqdm', type=bool, default=True, help='进度条')
    parser.add_argument('--train_dir', type=str, help='data dir',
                        default='/media/l/e6aa5997-4a1e-42e4-8782-83e2693751bd/city/data/tc_2021/suichang_round1_train_210120/')
    parser.add_argument('--log_dir', type=str, help='data dir',
                        default='/media/l/e6aa5997-4a1e-42e4-8782-83e2693751bd/city/logs',)
    parser.add_argument('--use_tensorboard', type=bool, default=True)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = get_args()
    args.log_dir = get_dir(filedir=args.log_dir, str='CITY_ECC')
    torch.backends.cudnn.enabled = True
    np.set_printoptions(precision=2)

    setproctitle.setproctitle('CITY_ECC')
    set_logger(os.path.join(args.log_dir, 'logfile.txt'))
    writer = SummaryWriter(args.log_dir + '/') if args.use_tensorboard else None
    get_gpuid(args.ngpus)
    filelst = []

    for name in os.listdir(args.train_dir):
        if name.endswith('.tif'):
            if os.path.exists(os.path.join(args.train_dir, name[:-4] + '.png')):
                filelst.append((os.path.join(args.train_dir, name), os.path.join(args.train_dir, name[:-4] + '.png')))
    datasets = ListDataSet(filelst)
    train_dataset, val_dataset = random_split(datasets, [datasets.__len__() // 5 * 4, datasets.__len__() - (datasets.__len__() // 5 * 4)])
    NET = get_eccnet(gpu_ids=args.ngpus, num_classes=10)
    # ckpt = torch.load('/media/l/e6aa5997-4a1e-42e4-8782-83e2693751bd/city/logs/CITY_ECC_21-02-04_3/E_54.ckpt')
    # NET.load_state_dict(ckpt)

    train_loader = DataLoader(train_dataset, batch_size=32 * args.ngpus, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=32 * args.ngpus, shuffle=True, num_workers=4)

    cri = Criterion(cri1=LossWrapper('FocalLoss'), cri2=LossWrapper('MultiClassDiceLoss')).cuda()
    opt = Adadelta(NET.parameters(), weight_decay=1e-5)
    # opt = SGD(NET.parameters(), lr=0.01, weight_decay=1e-5, nesterov=True, momentum=0.1)
    iou = IOUMetric(10)
    best_val_miou = 0.1
    best_val_acc_cls = 0.1
    for epoch in range(args.epochs):
        losses = []
        metrices = {}
        metrices['Epoch/epoch'] = epoch
        st = time()
        NET.train()
        for img, label in tqdm(train_loader, ncols=80, disable=not args.use_tqdm, desc='train'):
            pred = NET(img)
            loss = cri(pred, label.cuda())
            loss.backward()
            opt.step()
            opt.zero_grad()
            losses.append(loss.item())
            if isinstance(pred, list):
                pred = pred[0]

            iou.add_batch(torch.argmax(F.softmax(pred, dim=1), dim=1).cpu().numpy(), label.numpy())
        # print(iou.evaluate())
        metrices['TRAIN/LOSS'] = mean(losses)
        metrices['TRAIN/acc'], metrices['TRAIN/acc_cls'], _, metrices['TRAIN/miou'], _ = iou.evaluate()
        losses = []
        with torch.no_grad():
            NET.eval()
            for img, label in tqdm(val_loader, ncols=80, disable=not args.use_tqdm, desc='val  '):
                pred = NET(img)
                loss = cri(pred, label.cuda())
                losses.append(loss.item())
                if isinstance(pred, list):
                    pred = pred[0]
                iou.add_batch(torch.argmax(F.softmax(pred, dim=1), dim=1).cpu().numpy(), label.numpy())
        metrices['VAL/acc'], metrices['VAL/acc_cls'], _, metrices['VAL/miou'], _ = iou.evaluate()
        metrices['VAL/loss'] = mean(losses)
        metrices['Epoch/time'] = time() - st

        if best_val_miou < metrices['VAL/miou']:
            best_val_miou = metrices['VAL/miou']
            save_checkpoint(NET.state_dict(), os.path.join(args.log_dir, 'E_{}.ckpt'.format(epoch)))
        if best_val_acc_cls < metrices['VAL/acc_cls']:
            best_val_acc_cls = metrices['VAL/acc_cls']
            save_checkpoint(NET.state_dict(), os.path.join(args.log_dir, 'E_{}.ckpt'.format(epoch)))
        save_checkpoint(NET.state_dict(), os.path.join(args.log_dir, 'last.ckpt'))
        logging.info(' '.join(['{key}: {value:.3f}'.format(key=key, value=value) for key, value in metrices.items()]))
        for key, value in zip(metrices.keys(), metrices.values()):
            writer.add_scalar(key, value, epoch)
