from .deeplabs.modeling import deeplabv3plus_resnet101, deeplabv3plus_resnet50
from .ops import init_network


def GetMyDeepLabv3Plus(gpu_ids=1, ema=False, num_classes=10):
    # net = network.deeplabv3plus_resnet101(num_classes=1)
    net = deeplabv3plus_resnet50(num_classes=num_classes, pretrained_backbone=False)
    for param in net.parameters():
        param.requires_grad = True
    if ema:
        for param in net.parameters():
            param.detach_()
    return init_network(net, gpu_ids)


if __name__ == '__main__':
    model = GetMyDeepLabv3Plus()
    print(model)
