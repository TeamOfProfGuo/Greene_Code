import os
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import BatchNorm2d
from .basic import ConvBNReLU

__all__ = ['SpatialPath_drn']


class AttentionRefinementModule(nn.Module):
    def __init__(self, in_chan, out_chan, *args, **kwargs):
        super(AttentionRefinementModule, self).__init__()
        self.conv = ConvBNReLU(in_chan, out_chan, ks=3, stride=1, padding=1)
        self.conv_atten = nn.Sequential(nn.Conv2d(out_chan, out_chan, kernel_size= 1, bias=False),
                                        BatchNorm2d(out_chan))
        self.init_weight()

    def forward(self, x):
        feat = self.conv(x)
        atten = torch.mean(feat, dim=(2, 3), keepdim=True)   # [B, out_c, 1, 1]
        atten = self.conv_atten(atten)                       # [B, out_c, 1, 1]
        atten = atten.sigmoid()                              # [B, out_c, 1, 1]
        out = torch.mul(feat, atten)
        return out

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)


class SpatialPath_drn(nn.Module):
    """drn_c_26"""
    def __init__(self, root='./encoding/models/pretrain', sp_conv=None, *args, **kwargs):
        super(SpatialPath_drn, self).__init__()
        channels = (16, 32, 64, 128, 256, 512, 512, 512)
        layers = [1, 1, 2, 2, 2, 2, 1, 1]
        self.with_conv = sp_conv
        self.path = os.path.join(root, 'drn_c_26-ddedf421.pth')
        self.inplanes = channels[0]
        self.conv1 = nn.Conv2d(3, channels[0], kernel_size=7, stride=1, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(channels[0])
        self.relu = nn.ReLU(inplace=True)        # [B, 16, 480, 480]

        self.layer1 = self._make_layer(BasicBlock, channels[0], layers[0], stride=1)  # [B, 16, 480, 480]
        self.layer2 = self._make_layer(BasicBlock, channels[1], layers[1], stride=2)  # [B, 32, 240, 240]
        self.layer3 = self._make_layer(BasicBlock, channels[2], layers[2], stride=2)  # [B, 64, 120, 120]
        self.layer4 = self._make_layer(BasicBlock, channels[3], layers[3], stride=2)  # [B, 128, 60, 60]
        self.init_weight()

        self.d_layer0 = nn.Sequential(nn.Conv2d(1, channels[0], kernel_size=7, stride=1, padding=3, bias=False),
                                      nn.BatchNorm2d(channels[0]),
                                      nn.ReLU(inplace=True),)
        for i in range(1, 5):
            self.add_module('d_layer'+str(i), copy.deepcopy(self.__getattr__('layer'+str(i))))

        if self.with_conv=='cv1':
            self.conv_out = ConvBNReLU(128, 128, ks=1, stride=1, padding=0)
        elif self.with_conv=='cv3':
            self.conv_out = ConvBNReLU(128, 128, ks=3, stride=1, padding=0)

    def forward(self, x, d):
        x0 = self.relu(self.bn1(self.conv1(x)))
        d0 = self.d_layer0(d)
        # x0 = x0+d0
        x1 = self.layer1(x0)
        d1 = self.d_layer1(d0)
        x1 = x1+d1             # [B, 16, 480, 480]
        x2 = self.layer2(x1)
        d2 = self.d_layer2(d1)
        x2 = x2+d2             # [B, 32, 240, 240]
        x3 = self.layer3(x2)
        d3 = self.d_layer3(d2)
        x3 = x3+d3             # [B, 64, 120, 120]
        x4 = self.layer4(x3)
        d4 = self.d_layer4(d3)
        x4 = x4+d4             # [B, 128, 60, 60]

        if self.with_conv is not None:
            return self.conv_out(x4)
        else:
            return x4

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1, new_level=True, residual=True):
        assert dilation == 1 or dilation % 2 == 0
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion))

        layers = list()
        layers.append(block(
            self.inplanes, planes, stride, downsample,residual=residual,
            dilation=(1, 1) if dilation == 1 else (dilation // 2 if new_level else dilation, dilation) ))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, residual=residual, dilation=(dilation, dilation)))

        return nn.Sequential(*layers)

    def init_weight(self):
        pretrain_dict = torch.load(self.path)
        model_dict = self.state_dict()
        pretrained_dict = {k: v for k, v in pretrain_dict.items() if k in model_dict}
        # 2. overwrite entries in the existing state dict
        model_dict.update(pretrained_dict)
        # 3. load the new state dict
        self.load_state_dict(pretrained_dict)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, dilation=(1, 1), residual=True):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=dilation[0], bias=False, dilation=dilation[0])
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=dilation[1], bias=False, dilation=dilation[1])
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride
        self.residual = residual

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)
        if self.residual:
            out += residual
        out = self.relu(out)

        return out

