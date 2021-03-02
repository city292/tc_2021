import datetime
import logging
import os
import random
import shutil
import sys
import traceback
from argparse import ArgumentParser
from os.path import abspath, basename, dirname, join, isdir
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
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm
from time import time
try:
    from nets import LossWrapper
except:
    path = dirname(dirname(abspath(__file__)))
    sys.path.append(path)
    from nets import LossWrapper
finally:
    from nets import IOUMetric, get_eccnet, get_attu_net, get_enet, GetMyDeepLabv3Plus, GetMyDeepLab, get_CCNET_Model, get_CCUNET
    from nets import get_CBAM_U_Net, get_MyCCNET_Model
    from utils import ListDataSet, ListDataSetB3


def beijing(sec, what):
    beijing_time = datetime.datetime.now() + datetime.timedelta(hours=8)
    return beijing_time.timetuple()


def set_logger(logfile, on_NCloud=False):
    if on_NCloud:
        logging.Formatter.converter = beijing
    logger = logging.getLogger()  # 不加名称设置root logger
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

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
    memory_gpu = sorted(list(zip(memory_gpu, list(range(len(memory_gpu))))), key=lambda x: x[0])
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


def include_patterns(*patterns):
    def _ignore_patterns(pathname, names):
        ignore = set(name for name in names for pattern in patterns
                     if not name.endswith(pattern) and not isdir(join(pathname, name)))
        if '__pycache__' in names:
            ignore.add('__pycache__')
        return ignore

    return _ignore_patterns


def get_dir(filedir=None, nb_params=0, str=''):
    now = datetime.datetime.now()
    day = now.strftime('%y-%m-%d')

    for i in range(1000):
        tem = '%s_%s_%d' % (str, day, i)
        if os.path.exists(os.path.join(filedir, tem)):
            continue
        os.mkdir(os.path.join(filedir, tem))
        path = dirname(abspath(__file__))
        shutil.copytree(path, os.path.join(filedir, tem) + '/py', ignore=include_patterns('.py'))
        return os.path.join(filedir, tem)
    return None


def train(NET, train_loader, val_loader, cri, optimizer, scheduler, epochs, writer, args):
    iou = IOUMetric(10)
    best_val_miou = 0.1
    best_val_acc_cls = 0.1
    for epoch in range(epochs):
        losses = []
        metrices = {}
        # metrices['Epoch/epoch'] = epoch
        st = time()
        NET.train()
        batch_idx = 0
        for img, label in tqdm(train_loader, ncols=80, disable=not args.use_tqdm, desc='train'):
            pred = NET(img)
            loss = cri(pred, label.cuda())
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            losses.append(loss.item())
            if isinstance(pred, list):
                pred = pred[0]

            iou.add_batch(torch.argmax(F.softmax(pred, dim=1), dim=1).cpu().numpy(), label.numpy())
            scheduler.step(epoch + batch_idx / train_loader.__len__())
            batch_idx += 1
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
        metrices['lr/lr'] = optimizer.param_groups[-1]['lr']
        save_flag = False
        if best_val_miou < metrices['VAL/miou']:
            best_val_miou = metrices['VAL/miou']
            save_flag = True

        # if best_val_acc_cls < metrices['VAL/acc_cls']:
        #     best_val_acc_cls = metrices['VAL/acc_cls']
        #     save_flag = True
        ckpt = {'segnet': NET.state_dict(), 'args': args}
        if save_flag:
            save_checkpoint(ckpt, os.path.join(args.log_dir, 'E_{}.ckpt'.format(epoch)))
        save_checkpoint(ckpt, os.path.join(args.log_dir, 'last.ckpt'))
        logging.info(' '.join(['{key}: {value:.3f}'.format(key=key, value=value) for key, value in metrices.items()]))
        if writer is not None:
            for key, value in zip(metrices.keys(), metrices.values()):
                writer.add_scalar(key, value, epoch)


def print_options(args):
    message = ''
    message += '----------------- Options ---------------\n'
    for k, v in sorted(vars(args).items()):
        comment = ''
        message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
    message += '----------------- End -------------------'
    logging.info(message)


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
    parser.add_argument('--ngpus', type=int, default=2, help='GPU数')
    parser.add_argument('--epochs', type=int, default=500, help='最大迭代epoch数')
    parser.add_argument('--use_tqdm', type=bool, default=True, help='进度条')
    parser.add_argument('--train_dir', type=str, help='data dir',
                        default='/media/l/e6aa5997-4a1e-42e4-8782-83e2693751bd/city/data/tc_2021/suichang_round1_train_210120/')
    parser.add_argument('--log_dir', type=str, help='data dir',
                        default='/media/l/e6aa5997-4a1e-42e4-8782-83e2693751bd/city/logs',)
    parser.add_argument('--use_tensorboard', type=bool, default=True)
    parser.add_argument('--segname', type=str, default='enet', help='分割网络模型',
                        choices=['attu_net', 'deeplabv3plus', 'deeplabv3', 'enet', 'ccnet', 'eccnet', 'ccunet', 'CBAMUNET', 'myccnet'])
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--img_ch', type=int, default=3, choices=[3, 4])
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = get_args()
    args.log_dir = get_dir(filedir=args.log_dir, str='CITY_tc_' + args.segname)
    torch.backends.cudnn.enabled = True
    np.set_printoptions(precision=2)

    setproctitle.setproctitle('CITY_' + args.segname + '_' + args.log_dir[-7:])
    set_logger(os.path.join(args.log_dir, 'logfile.txt'))
    writer = SummaryWriter(args.log_dir + '/') if args.use_tensorboard else None
    get_gpuid(args.ngpus)
    print_options(args)
    filelst = []
    try:
        from aicloud.engine import AlgorithmEngine
        ae = AlgorithmEngine()
        args.on_Ncloud = True
    except:
        args.on_Ncloud = False
    if args.on_Ncloud:
        msg2 = ae.create_training_task(os.path.basename(args.log_dir), '/root/work/logs')
        logging.info(msg2)
        args.use_tensorboard = True
        logging.info('On NCloud')
    else:
        logging.info('Not on NCloud')
    for name in os.listdir(args.train_dir):
        if name.endswith('.tif'):
            if os.path.exists(os.path.join(args.train_dir, name[:-4] + '.png')):
                filelst.append((os.path.join(args.train_dir, name), os.path.join(args.train_dir, name[:-4] + '.png')))
    if args.img_ch == 4:
        datasets = ListDataSet(filelst)
    else:
        datasets = ListDataSetB3(filelst)
    train_dataset, val_dataset = random_split(datasets, [datasets.__len__() // 5 * 4, datasets.__len__() - (datasets.__len__() // 5 * 4)])
    # NET = get_eccnet(gpu_ids=args.ngpus, num_classes=10)
    # NET = get_attu_net(gpu_ids=args.ngpus, num_classes=10)
    # ckpt = torch.load('/media/l/e6aa5997-4a1e-42e4-8782-83e2693751bd/city/logs/CITY_ECC_21-02-04_3/E_54.ckpt')
    # NET.load_state_dict(ckpt)
    if args.segname == 'attu_net':
        NET = get_attu_net(gpu_ids=args.ngpus, num_classes=10, img_ch=args.img_ch)
    elif args.segname == 'eccnet':
        NET = get_eccnet(gpu_ids=args.ngpus, num_classes=10)
    elif args.segname == 'enet':
        NET = get_enet(gpu_ids=args.ngpus, num_classes=10)
    elif args.segname == 'deeplabv3plus':
        NET = GetMyDeepLabv3Plus(gpu_ids=args.ngpus, num_classes=10)
    elif args.segname == 'deeplabv3':
        NET = GetMyDeepLab(gpu_ids=args.ngpus, num_classes=10)
    elif args.segname == 'ccnet':
        NET = get_CCNET_Model(gpu_ids=args.ngpus, num_classes=10, img_ch=args.img_ch)
    elif args.segname == 'ccunet':
        NET = get_CCUNET(gpu_ids=args.ngpus, num_classes=10, img_ch=args.img_ch)
    elif args.segname == 'CBAMUNET':
        NET = get_CBAM_U_Net(gpu_ids=args.ngpus, num_classes=10)
    elif args.segname == 'myccnet':
        NET = get_MyCCNET_Model(gpu_ids=args.ngpus, num_classes=10)

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=True, num_workers=4)

    cri = Criterion(cri1=LossWrapper('MultiFocalLoss'), cri2=LossWrapper('MultiClassDiceLoss')).cuda()
    # optimizer = Adam(NET.parameters(), weight_decay=1e-5)
    # opt = Adadelta(NET.parameters(), weight_decay=1e-5)
    optimizer = SGD(NET.parameters(), lr=1e-2, momentum=0.9, weight_decay=5e-4)
    # optimizer = optim.SGD(model.parameters(), lr=1e-2, momentum=momentum, weight_decay=weight_decay)
    # scheduler = StepLR(optimizer, step_size=30, gamma=0.2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=3, T_mult=2, eta_min=1e-5, last_epoch=-1)
    # opt = SGD(NET.parameters(), lr=0.01, weight_decay=1e-5, nesterov=True, momentum=0.1)
    train(NET, train_loader, val_loader, cri, optimizer, scheduler, args.epochs, writer, args)
