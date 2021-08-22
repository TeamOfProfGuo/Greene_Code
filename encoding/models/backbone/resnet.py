import os
import torch
import torch.nn as nn
import torchvision.models as models
from .resnetc import resnet50

__all__ = ['get_backbone']


def get_backbone(backbone='resnet18', input_dim=3, pretrained=True, root='../../encoding/models/pretrain'):
    assert input_dim in (1, 3, 4)

    if backbone == 'resnet18':
        model = models.resnet18(pretrained=False)
        fname = 'resnet18-5c106cde.pth'
    elif backbone == 'resnet50':
        model = models.resnet50(pretrained=False)
        fname = 'resnet50-19c8e357.pth'
    elif backbone == 'resnet50c':
        model = resnet50(pretrained=False)
        fname = 'resnet50_v2.pth'

    f_path = os.path.join(root, fname)
    if pretrained:
        if not os.path.exists(f_path):
            raise FileNotFoundError('The pretrained model {} cannot be found'.format(f_path))
        model.load_state_dict(torch.load(f_path), strict=False)
    if input_dim != 3:
        model.conv1 = nn.Conv2d(input_dim, 64, kernel_size=7, stride=2, padding=3, bias=False)

    return model



def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')

        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride, dilation=dilation)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes, dilation=dilation)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


def dilated_resnet18(dilation=2):
    if dilation == 2:
        dilation_list = [False, False, True]
    elif dilation == 4:
        dilation_list = [False, True, True]
    model = models.ResNet(BasicBlock, [2, 2, 2, 2], replace_stride_with_dilation=dilation_list)
    return model
