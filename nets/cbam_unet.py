import os
import sys
from os.path import dirname, abspath, basename
import setproctitle
from torch import nn
import torch
from .ops import init_network
from .CBAM import CBAM_Module
from .attention_unet import DoubleConv, up_conv


class CBAM_U_Net(nn.Module):
    def __init__(self, img_ch=3, output_ch=1, base_features=32):
        super(CBAM_U_Net, self).__init__()

        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Conv1 = DoubleConv(ch_in=img_ch, ch_out=base_features)
        self.Conv2 = DoubleConv(ch_in=base_features, ch_out=base_features * 2)
        self.Conv3 = DoubleConv(ch_in=base_features * 2, ch_out=base_features * 4)
        self.Conv4 = DoubleConv(ch_in=base_features * 4, ch_out=base_features * 8)
        self.Conv5 = DoubleConv(ch_in=base_features * 8, ch_out=base_features * 16)

        self.Up5 = up_conv(ch_in=base_features * 16, ch_out=base_features * 8)
        self.Att5 = CBAM_Module(channels=base_features * 16)
        self.Up_conv5 = DoubleConv(ch_in=base_features * 16, ch_out=base_features * 8)

        self.Up4 = up_conv(ch_in=base_features * 8, ch_out=base_features * 4)
        self.Att4 = CBAM_Module(channels=base_features * 8)
        self.Up_conv4 = DoubleConv(ch_in=base_features * 8, ch_out=base_features * 4)

        self.Up3 = up_conv(ch_in=base_features * 4, ch_out=base_features * 2)
        self.Att3 = CBAM_Module(channels=base_features * 4)
        self.Up_conv3 = DoubleConv(ch_in=base_features * 4, ch_out=base_features * 2)

        self.Up2 = up_conv(ch_in=base_features * 2, ch_out=base_features)
        self.Att2 = CBAM_Module(channels=base_features * 2)
        self.Up_conv2 = DoubleConv(ch_in=base_features * 2, ch_out=base_features)

        self.Att1 = CBAM_Module(channels=base_features * 2)
        self.Conv_1x1 = nn.Conv2d(base_features, output_ch, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        # encoding path
        x1 = self.Conv1(x)

        x2 = self.Maxpool(x1)
        x2 = self.Conv2(x2)

        x3 = self.Maxpool(x2)
        x3 = self.Conv3(x3)

        x4 = self.Maxpool(x3)
        x4 = self.Conv4(x4)

        x5 = self.Maxpool(x4)
        x5 = self.Conv5(x5)

        # decoding + concat path
        d5 = self.Up5(x5)
        d5 = torch.cat((x4, d5), dim=1)
        d5 = self.Att5(d5)

        d5 = self.Up_conv5(d5)

        d4 = self.Up4(d5)
        d4 = torch.cat((x3, d4), dim=1)
        d4 = self.Att4(d4)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.Att3(d3)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        d2 = torch.cat((x1, d2), dim=1)
        d2 = self.Att2(d2)
        d2 = self.Up_conv2(d2)

        # d1 = self.Att1(d2)
        d1 = self.Conv_1x1(d2)

        return d1


def get_CBAM_U_Net(gpu_ids=1, ema=False, num_classes=1):
    net = CBAM_U_Net(img_ch=4, output_ch=num_classes)
    if ema:
        for param in net.parameters():
            param.detach_()
    return init_network(net, gpu_ids)
