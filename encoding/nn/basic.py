# encoding:utf-8

import torch
import torch.nn as nn
from torch.nn import functional as F

__all__ = ['BasicBlock', 'FuseBlock', 'ConvBNReLU', 'CenterBlock', 'FCNHead']


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


class IRB_Block(nn.Module):
    def __init__(self, in_feats, out_feats=None, act='idt', expand_ratio=6):
        super().__init__()
        mid_feats = round(in_feats * expand_ratio)
        out_feats = in_feats if out_feats is None else out_feats
        act_layer = nn.Identity() if act == 'idt' else nn.ReLU6(inplace=True)
        self.idt = (in_feats == out_feats)
        self.irb = nn.Sequential(
                # point-wise conv
                nn.Conv2d(in_feats, mid_feats, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(mid_feats),
                nn.ReLU6(inplace=True),
                # depth-wise conv
                nn.Conv2d(mid_feats, mid_feats, kernel_size=3, stride=1, padding=1, groups=mid_feats, bias=False),
                nn.BatchNorm2d(mid_feats),
                nn.ReLU6(inplace=True),
                # point-wise conv
                nn.Conv2d(mid_feats, out_feats, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(out_feats),
                act_layer
            )

    def forward(self, x):
        return (x + self.irb(x)) if self.idt else self.irb(x)


class LearnedUpUnit(nn.Module):
    def __init__(self, in_feats):
        super().__init__()
        self.up = nn.UpsamplingNearest2d(scale_factor=2)
        self.dep_conv = nn.Conv2d(in_feats, in_feats, kernel_size=3, stride=1, padding=1, groups=in_feats, bias=False)

    def forward(self, x):
        x = self.up(x)
        x = self.dep_conv(x)
        return x


class ConvBNReLU(nn.Module):

    def __init__(self, in_chan, out_chan, ks=3, stride=1, padding=1, *args, **kwargs):
        super(ConvBNReLU, self).__init__()
        self.conv = nn.Conv2d(in_chan, out_chan, kernel_size = ks, stride = stride, padding = padding, bias = False)
        self.bn = nn.BatchNorm2d(out_chan)
        self.relu = nn.ReLU(inplace=True)
        self.init_weight()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)


class UpSample(nn.Module):

    def __init__(self, n_chan, factor=2):
        super(UpSample, self).__init__()
        out_chan = n_chan * factor * factor
        self.proj = nn.Conv2d(n_chan, out_chan, 1, 1, 0)
        self.up = nn.PixelShuffle(factor)
        self.init_weight()

    def forward(self, x):
        feat = self.proj(x)
        feat = self.up(feat)
        return feat

    def init_weight(self):
        nn.init.xavier_normal_(self.proj.weight, gain=1.)


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




class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class GlobalPooling(nn.Module):
    def __init__(self, in_channels, out_channels, norm_layer, up_kwargs):
        super(GlobalPooling, self).__init__()
        self._up_kwargs = up_kwargs
        self.gap = nn.Sequential(nn.AdaptiveAvgPool2d(1),
                                 nn.Conv2d(in_channels, out_channels, 1, bias=False),
                                 norm_layer(out_channels),
                                 nn.ReLU(True))

    def forward(self, x):
        _, _, h, w = x.size()
        pool = self.gap(x)
        return F.interpolate(pool, (h, w), **self._up_kwargs)


class ConcurrentModule(nn.ModuleList):
    r"""Feed to a list of modules concurrently. The outputs of the layers are concatenated at channel dimension.

    Args: modules (iterable, optional): an iterable of modules to add
    """
    def __init__(self, modules=None):
        super(ConcurrentModule, self).__init__(modules)

    def forward(self, x):
        outputs = []
        for layer in self:
            outputs.append(layer(x))
        return torch.cat(outputs, 1)


class FCNHead(nn.Module):
    def __init__(self, in_channels, out_channels, norm_layer=nn.BatchNorm2d, up_kwargs={'mode': 'bilinear', 'align_corners': True}, with_global=False):
        super(FCNHead, self).__init__()
        inter_channels = in_channels // 4
        self._up_kwargs = up_kwargs
        if with_global:
            self.conv5 = nn.Sequential(nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
                                       norm_layer(inter_channels),
                                       nn.ReLU(),
                                       ConcurrentModule([
                                           Identity(),
                                           GlobalPooling(inter_channels, inter_channels,
                                                         norm_layer, self._up_kwargs),
                                       ]),
                                       nn.Dropout(0.1, False),
                                       nn.Conv2d(2 * inter_channels, out_channels, 1))
        else:
            self.conv5 = nn.Sequential(nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
                                       norm_layer(inter_channels),
                                       nn.ReLU(),
                                       nn.Dropout(0.1, False),
                                       nn.Conv2d(inter_channels, out_channels, 1))

    def forward(self, x):
        return self.conv5(x)

