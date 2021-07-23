# encoding:utf-8

import torch
import torch.nn as nn

__all__ = ['BasicBlock', 'FuseBlock', 'CenterBlock']


class BasicBlock(nn.Module):
    def __init__(self, inplanes, planes, upsample=False):
        super(BasicBlock, self).__init__()
        self.upsample = upsample
        self.CBR1 = nn.Sequential(nn.Conv2d(inplanes, planes, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.BatchNorm2d(planes),
                                  nn.ReLU(inplace=True))

        if not upsample:
            self.conv2 = nn.Sequential(nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False),
                                       nn.BatchNorm2d(planes))
            if planes!= inplanes:
                self.up = nn.Sequential(nn.Conv2d(inplanes, planes, kernel_size=1, stride=1, bias=False),
                                        nn.BatchNorm2d(planes))
            else:
                self.up = None
        else:
            self.conv2=nn.Sequential(nn.ConvTranspose2d(planes, planes, kernel_size=2, stride=2, padding=0, bias=False),
                                     nn.BatchNorm2d(planes))
            self.up = nn.Sequential(nn.ConvTranspose2d(inplanes, planes, kernel_size=2, stride=2, padding=0, bias=False),
                                    nn.BatchNorm2d(planes))
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        if self.up is not None:
            identity = self.up(x)
        else:
            identity = x

        out = self.CBR1(x)
        out = self.conv2(out)
        out += identity
        out = self.relu(out)

        return out


class FuseBlock(nn.Module):
    def __init__(self, in_chs, out_ch):
        # in_chs=[g_ch, x_ch]: g_ch 为深层特征的channel数， x_ch is num of channels for features from encoder
        super().__init__()
        assert len(in_chs)==2, 'provide num of channels for both inputs'
        g_ch, x_ch = in_chs
        self.g_rcu0 = BasicBlock(g_ch, g_ch)
        self.x_rcu0 = BasicBlock(x_ch, x_ch)

        self.g_rcu1 = BasicBlock(g_ch, out_ch, upsample=True)
        self.x_rcu1 = BasicBlock(x_ch, out_ch, upsample=False)

        self.g_conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False)
        self.x_conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g = self.g_rcu1(self.g_rcu0(g))
        x = self.x_rcu1(self.x_rcu0(x))
        g = self.g_conv2(g)
        x = self.x_conv2(x)
        out = self.relu(g+x)
        return out


class CenterBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.rcu = nn.Sequential(BasicBlock(in_ch, in_ch),
                                 BasicBlock(in_ch, out_ch))

    def forward(self, g):
        out = self.rcu(g)
        return out


