import torch
import torch.nn.functional as F
from torch.nn import Sigmoid
import numpy as np
import torch.nn as nn


class LossWrapper(nn.Module):
    def __init__(self, name=None):
        super(LossWrapper, self).__init__()

        if name == 'FocalLoss':
            self.lossF = FocalLoss()
        elif name == 'OHEMFocalLoss':
            self.lossF = OHEMFocalLoss()
        elif name == 'BCE':
            self.lossF = nn.BCEWithLogitsLoss()
        elif name == 'DiceLoss':
            self.lossF = DiceLoss()
        elif name == 'BoundaryLoss':
            self.lossF = BoundaryLoss()

    def forward(self, preds, labels):
        h, w = labels.size(1), labels.size(2)

        if isinstance(preds, list):

            loss1 = self.lossF(preds[0], labels)
            loss2 = self.lossF(preds[1], labels)

            loss = loss1 * 0.8 + loss2 * 0.2

            return loss
        else:
            return self.lossF(preds, labels)


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2, weight=None, ignore_index=None):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.weight = weight
        self.ignore_index = ignore_index
        self.bce_fn = nn.BCEWithLogitsLoss(weight=self.weight)

    def forward(self, preds, labels):
        if self.ignore_index is not None:
            mask = labels != self.ignore_index
            labels = labels[mask]
            preds = preds[mask]
        preds = preds.contiguous().view(-1)
        labels = labels.contiguous().view(-1)
        logpt = -self.bce_fn(preds, labels)
        pt = torch.exp(logpt)
        loss = -((1 - pt) ** self.gamma) * self.alpha * logpt
        return loss


def diceCoeff(pred, gt, smooth=1e-5, activation='sigmoid'):
    r""" computational formula：
        dice = (2 * (pred ∩ gt)) / (pred ∪ gt)
    """

    if activation is None or activation == "none":
        def activation_fn(x): return x
    elif activation == "sigmoid":
        activation_fn = nn.Sigmoid()
    elif activation == "softmax2d":
        activation_fn = nn.Softmax2d()
    else:
        raise NotImplementedError("Activation implemented for sigmoid and softmax2d activation function operation")

    pred = activation_fn(pred)

    N = gt.size(0)
    pred_flat = pred.view(N, -1)
    gt_flat = gt.view(N, -1)

    intersection = (pred_flat * gt_flat).sum(1)
    unionset = pred_flat.sum(1) + gt_flat.sum(1)
    loss = (2 * intersection + smooth) / (unionset + smooth)

    return loss.sum() / N


def diceCoeffv2(pred, gt, eps=1e-5, activation='sigmoid'):
    r""" computational formula：
        dice = (2 * tp) / (2 * tp + fp + fn)
    """

    if activation is None or activation == "none":
        def activation_fn(x): return x
    elif activation == "sigmoid":
        activation_fn = nn.Sigmoid()
    elif activation == "softmax2d":
        activation_fn = nn.Softmax2d()
    else:
        raise NotImplementedError("Activation implemented for sigmoid and softmax2d activation function operation")

    pred = activation_fn(pred)

    N = gt.size(0)
    pred_flat = pred.view(N, -1)
    gt_flat = gt.view(N, -1)

    tp = torch.sum(gt_flat * pred_flat, dim=1)
    fp = torch.sum(pred_flat, dim=1) - tp
    fn = torch.sum(gt_flat, dim=1) - tp
    loss = (2 * tp + eps) / (2 * tp + fp + fn + eps)
    return loss.sum() / N


class DiceLoss(nn.Module):
    __name__ = 'dice_loss'

    def __init__(self, activation='sigmoid'):
        super(DiceLoss, self).__init__()
        self.activation = activation

    def forward(self, y_pr, y_gt):
        return 1 - diceCoeffv2(y_pr, y_gt, activation=self.activation)


class OHEMFocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2, weight=None, ignore_index=255, OHEM_percent=0.3):
        super(OHEMFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.weight = weight
        self.ignore_index = ignore_index
        # self.bce_fn = nn.BCEWithLogitsLoss(weight=self.weight)
        self.OHEM_percent = OHEM_percent

    def forward(self, output, target):
        output = output.contiguous().view(-1)
        target = target.contiguous().view(-1)
        max_val = (-output).clamp(min=0)
        loss = output - output * target + max_val + ((-max_val).exp() + (-output - max_val).exp()).log()
        # This formula gives us the log sigmoid of 1-p if y is 0 and of p if y is 1
        invprobs = F.logsigmoid(-output * (target * 2 - 1))
        focal_loss = self.alpha * (invprobs * self.gamma).exp() * loss
        # Online Hard Example Mining: top x% losses (pixel-wise).
        # Refer to http://www.robots.ox.ac.uk/~tvg/publications/2017/0026.pdf
        OHEM, _ = focal_loss.topk(k=int(self.OHEM_percent * [*focal_loss.shape][0]))
        return OHEM.mean()
        # return loss


def ohem_focal_loss(self, output, target, alpha, gamma, OHEM_percent):
    output = output.contiguous().view(-1)
    target = target.contiguous().view(-1)

    max_val = (-output).clamp(min=0)
    loss = output - output * target + max_val + ((-max_val).exp() + (-output - max_val).exp()).log()

    # This formula gives us the log sigmoid of 1-p if y is 0 and of p if y is 1
    invprobs = F.logsigmoid(-output * (target * 2 - 1))
    focal_loss = alpha * (invprobs * gamma).exp() * loss

    # Online Hard Example Mining: top x% losses (pixel-wise). Refer to http://www.robots.ox.ac.uk/~tvg/publications/2017/0026.pdf
    OHEM, _ = focal_loss.topk(k=int(OHEM_percent * [*focal_loss.shape][0]))
    return OHEM.mean()


def lovasz_grad(gt_sorted):
    """
    Computes gradient of the Lovasz extension w.r.t sorted errors
    See Alg. 1 in paper
    """
    p = len(gt_sorted)
    gts = gt_sorted.sum()
    intersection = gts - gt_sorted.float().cumsum(0)
    union = gts + (1 - gt_sorted).float().cumsum(0)
    jaccard = 1. - intersection / union
    if p > 1:  # cover 1-pixel case
        jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]
    return jaccard


def iou_pytorch(outputs: torch.Tensor, labels: torch.Tensor):
    # You can comment out this line if you are passing tensors of equal shape
    # But if you are passing output from UNet or something it will most probably
    # be with the BATCH x 1 x H x W shape
    SMOOTH = 1e-6
    # outputs = outputs.squeeze(1)  # BATCH x 1 x H x W => BATCH x H x W
    outputs = torch.round(Sigmoid()(outputs)).int()

    labels = labels.int()
    # print(outputs.sum(), labels.sum())
    intersection = (outputs & labels).float().sum()  # Will be zero if Truth=0 or Prediction=0

    union = (outputs | labels).float().sum()  # Will be zero if both are 0
    # print(intersection, union)
    iou = (intersection + SMOOTH) / (union + SMOOTH)  # We smooth our devision to avoid 0/0

    # thresholded = torch.clamp(20 * (iou - 0.5), 0, 10).ceil() / 10  # This is equal to comparing with thresolds
    # return thresholded.mean()
    return iou  # Or thresholded.mean() if you are interested in average across the batch


def IoU(inputs, targets, smooth=1):
    # 把inputs，targets转成cpu再detach，这样就不会占用GPU资源。
    inputs = inputs.cpu().detach()
    targets = targets.cpu().detach()
    # 要对input进行threshold，让他变成（0，1）组成的。
    inputs = torch.round(inputs).view(-1)
    targets = torch.round(targets).view(-1)
    # intersection is equivalent to True Positive count
    # union is the mutually inclusive area of all labels & predictions
    intersection = (inputs * targets).sum()
    total = (inputs + targets).sum()
    union = total - intersection

    IoU = (intersection + smooth) / (union + smooth)
    return IoU.numpy()


def one_hot(label, n_classes, requires_grad=True):
    """Return One Hot Label"""
    device = label.device

    one_hot_label = torch.eye(
        n_classes, device=device, requires_grad=requires_grad)[label]
    one_hot_label = one_hot_label.transpose(1, 3).transpose(2, 3)

    return one_hot_label


class BoundaryLoss(nn.Module):
    """Boundary Loss proposed in:
    Alexey Bokhovkin et al., Boundary Loss for Remote Sensing Imagery Semantic Segmentation
    https://arxiv.org/abs/1905.07852
    """

    def __init__(self, theta0=3, theta=5):
        super().__init__()

        self.theta0 = theta0
        self.theta = theta

    def forward(self, pred, gt):
        """
        Input:
            - pred: the output from model (before softmax)
                    shape (N, C, H, W)
            - gt: ground truth map
                    shape (N, H, w)
        Return:
            - boundary loss, averaged over mini-bathc
        """

        n, c, _, _ = pred.shape

        # softmax so that predicted map can be distributed in [0, 1]
        if c == 1:
            pred = torch.nn.Sigmoid()(pred)       # c == 1
            one_hot_gt = torch.unsqueeze(gt, 1)
        else:
            pred = torch.nn.Softmax()(pred)     # c > 1
            one_hot_gt = one_hot(gt, c)

        # boundary map
        gt_b = F.max_pool2d(
            1 - one_hot_gt, kernel_size=self.theta0, stride=1, padding=(self.theta0 - 1) // 2)
        gt_b -= 1 - one_hot_gt

        pred_b = F.max_pool2d(
            1 - pred, kernel_size=self.theta0, stride=1, padding=(self.theta0 - 1) // 2)
        pred_b -= 1 - pred

        # extended boundary map
        gt_b_ext = F.max_pool2d(
            gt_b, kernel_size=self.theta, stride=1, padding=(self.theta - 1) // 2)

        pred_b_ext = F.max_pool2d(
            pred_b, kernel_size=self.theta, stride=1, padding=(self.theta - 1) // 2)

        # reshape
        gt_b = gt_b.view(n, c, -1)
        pred_b = pred_b.view(n, c, -1)
        gt_b_ext = gt_b_ext.view(n, c, -1)
        pred_b_ext = pred_b_ext.view(n, c, -1)

        # Precision, Recall
        P = torch.sum(pred_b * gt_b_ext, dim=2) / (torch.sum(pred_b, dim=2) + 1e-7)
        R = torch.sum(pred_b_ext * gt_b, dim=2) / (torch.sum(gt_b, dim=2) + 1e-7)

        # Boundary F1 Score
        BF1 = 2 * P * R / (P + R + 1e-7)

        # summing BF1 Score for each class and average over mini-batch
        loss = torch.mean(1 - BF1)

        return loss

# class SurfaceLoss():
#     def __init__(self, **kwargs):
#         # Self.idc is used to filter out some classes of the target mask. Use fancy indexing
#         self.idc: List[int] = kwargs["idc"]
#         print(f"Initialized {self.__class__.__name__} with {kwargs}")

#     def __call__(self, probs: Tensor, dist_maps: Tensor, _: Tensor) -> Tensor:
#         assert simplex(probs)
#         assert not one_hot(dist_maps)

#         pc = probs[:, self.idc, ...].type(torch.float32)
#         dc = dist_maps[:, self.idc, ...].type(torch.float32)

#         multipled = einsum("bcwh,bcwh->bcwh", pc, dc)

#         loss = multipled.mean()


#         return loss
if __name__ == '__main__':
    criterion1 = OHEMFocalLoss()
    # criterion2 = nn.BCEWithLogitsLoss()
    criterion2 = DiceLoss()
    print(str(criterion1))
