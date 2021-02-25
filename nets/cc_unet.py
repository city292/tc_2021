import os
import sys
from os.path import dirname, abspath, basename
import setproctitle
from torch import nn
import torch
from .ops import init_network
from .attention_unet import DoubleConv, up_conv
from .CCNet import CrissCrossAttention


class CCUNET(nn.Module):
    def __init__(self, img_ch=3, output_ch=1, base_features=32):
        super(CCUNET, self).__init__()

        self.Conv1 = DoubleConv(ch_in=img_ch, ch_out=base_features)

        self.cca1 = CrissCrossAttention(base_features)
        self.cca2 = CrissCrossAttention(base_features)
        # self.cca3 = CrissCrossAttention(base_features)
        # self.cca4 = CrissCrossAttention(base_features)
        self.Conv2 = DoubleConv(ch_in=base_features, ch_out=base_features)
        self.Conv_1x1 = nn.Conv2d(base_features, output_ch, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        # encoding path
        x = self.Conv1(x)
        x = self.cca1(x)
        x = self.cca2(x)
        # x = self.cca3(x)
        # x = self.cca4(x)
        x = self.Conv2(x)

        x = self.Conv_1x1(x)

        return x


def get_CCUNET(gpu_ids=1, ema=False, num_classes=10):
    net = CCUNET(img_ch=4, output_ch=num_classes)
    if ema:
        for param in net.parameters():
            param.detach_()
    return init_network(net, gpu_ids)
