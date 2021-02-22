import os
import sys
from os.path import dirname, abspath, basename
import setproctitle
from torch import nn
import torch
from .ops import init_network


class AttU_Net(nn.Module):
    def __init__(self, img_ch=3, output_ch=1, base_features=32):
        super(AttU_Net, self).__init__()

        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Conv1 = DoubleConv(ch_in=img_ch, ch_out=base_features)
        self.Conv2 = DoubleConv(ch_in=base_features, ch_out=base_features * 2)
        self.Conv3 = DoubleConv(ch_in=base_features * 2, ch_out=base_features * 4)
        self.Conv4 = DoubleConv(ch_in=base_features * 4, ch_out=base_features * 8)
        self.Conv5 = DoubleConv(ch_in=base_features * 8, ch_out=base_features * 16)

        self.Up5 = up_conv(ch_in=base_features * 16, ch_out=base_features * 8)
        self.Att5 = Attention_block(F_g=base_features * 8, F_l=base_features * 8, F_int=base_features * 4)
        self.Up_conv5 = DoubleConv(ch_in=base_features * 16, ch_out=base_features * 8)

        self.Up4 = up_conv(ch_in=base_features * 8, ch_out=base_features * 4)
        self.Att4 = Attention_block(F_g=base_features * 4, F_l=base_features * 4, F_int=base_features * 2)
        self.Up_conv4 = DoubleConv(ch_in=base_features * 8, ch_out=base_features * 4)

        self.Up3 = up_conv(ch_in=base_features * 4, ch_out=base_features * 2)
        self.Att3 = Attention_block(F_g=base_features * 2, F_l=base_features * 2, F_int=base_features)
        self.Up_conv3 = DoubleConv(ch_in=base_features * 4, ch_out=base_features * 2)

        self.Up2 = up_conv(ch_in=base_features * 2, ch_out=base_features)
        self.Att2 = Attention_block(F_g=base_features, F_l=base_features, F_int=base_features // 2)
        self.Up_conv2 = DoubleConv(ch_in=base_features * 2, ch_out=base_features)

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
        x4 = self.Att5(g=d5, x=x4)
        d5 = torch.cat((x4, d5), dim=1)
        d5 = self.Up_conv5(d5)

        d4 = self.Up4(d5)
        x3 = self.Att4(g=d4, x=x3)
        d4 = torch.cat((x3, d4), dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        x2 = self.Att3(g=d3, x=x2)
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        x1 = self.Att2(g=d2, x=x1)
        d2 = torch.cat((x1, d2), dim=1)
        d2 = self.Up_conv2(d2)

        d1 = self.Conv_1x1(d2)

        return d1


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, ch_in, ch_out, ch_mid=None):
        super(DoubleConv, self).__init__()
        if not ch_mid:
            ch_mid = ch_in
        self.double_conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_mid, kernel_size=3, padding=1),
            nn.BatchNorm2d(ch_mid),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_mid, ch_out, kernel_size=3, padding=1),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class up_conv(nn.Module):
    def __init__(self, ch_in=1024, ch_out=512):
        super(up_conv, self).__init__()
        self.up = nn.ConvTranspose2d(ch_in, ch_out, kernel_size=2, stride=2)

    def forward(self, x):
        return self.up(x)


class Attention_block(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(Attention_block, self).__init__()
        self.W_g = nn.Sequential(nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
                                 nn.BatchNorm2d(F_int))

        self.W_x = nn.Sequential(nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
                                 nn.BatchNorm2d(F_int))

        self.psi = nn.Sequential(nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
                                 nn.BatchNorm2d(1),
                                 nn.Sigmoid())
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)

        return x * psi


def get_attu_net(gpu_ids=1, ema=False, num_classes=1):
    net = AttU_Net(img_ch=4, output_ch=10)
    if ema:
        for param in net.parameters():
            param.detach_()
    return init_network(net, gpu_ids)
