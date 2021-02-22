import torch.nn as nn
from .deeplabs.modeling import deeplabv3_resnet50
from .ops import init_network

# from .ops import init_network


class MyDeepLab(nn.Module):
    def __init__(self):
        super(MyDeepLab, self).__init__()
        self.deeplabv3 = deeplabv3_resnet50(pretrained_backbone=False, num_classes=10)
        self.__name__ = 'DEEPLABV3'

    def forward(self, x):

        outputs = self.deeplabv3(x)
        return outputs


def GetMyDeepLab(gpu_ids=1, ema=False, num_classes=10):
    net = MyDeepLab()
    if ema:
        for param in net.parameters():
            param.detach_()
    return init_network(net, gpu_ids)


if __name__ == '__main__':
    deeplabv3 = MyDeepLab()()
    print(deeplabv3)
    # print(net)
