#!/usr/bin/python
# -*- encoding: utf-8 -*-

import os
import torch
import torch.nn as nn
from ..backbone import get_resnet18
import torch.nn.functional as F
from torch.nn import BatchNorm2d
from ...nn import ConvBNReLU

__all__ = ['BiSeDNet', 'get_bised']


class BiSeNetOutput(nn.Module):

    def __init__(self, in_chan, mid_chan, n_classes, up_factor=32, *args, **kwargs):
        super(BiSeNetOutput, self).__init__()
        self.up_factor = up_factor
        out_chan = n_classes
        self.conv = ConvBNReLU(in_chan, mid_chan, ks=3, stride=1, padding=1)
        self.conv_out = nn.Conv2d(mid_chan, out_chan, kernel_size=1, bias=True)
        self.up = nn.Upsample(scale_factor=up_factor, mode='bilinear', align_corners=False)
        self.init_weight()

    def forward(self, x):
        x = self.conv(x)
        x = self.conv_out(x)
        x = self.up(x)
        return x

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)

    def get_params(self):
        wd_params, nowd_params = [], []
        for name, module in self.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                wd_params.append(module.weight)
                if not module.bias is None:
                    nowd_params.append(module.bias)
            elif isinstance(module, nn.modules.batchnorm._BatchNorm):
                nowd_params += list(module.parameters())
        return wd_params, nowd_params


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


class ContextPath(nn.Module):
    def __init__(self, root='./encoding/models/pretrain', *args, **kwargs):
        super(ContextPath, self).__init__()
        self.resnet = get_resnet18(input_dim=3, f_path=os.path.join(root, 'resnet18-5c106cde.pth'))
        self.arm16 = AttentionRefinementModule(256, 128)
        self.arm32 = AttentionRefinementModule(512, 128)
        self.conv_head32 = ConvBNReLU(128, 128, ks=3, stride=1, padding=1)
        self.conv_head16 = ConvBNReLU(128, 128, ks=3, stride=1, padding=1)
        self.conv_avg = ConvBNReLU(512, 128, ks=1, stride=1, padding=0)
        self.up32 = nn.Upsample(scale_factor=2.)
        self.up16 = nn.Upsample(scale_factor=2.)

        self.init_weight()

    def forward(self, x):
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        feat4 = self.resnet.layer1(x)
        feat8 = self.resnet.layer2(feat4)
        feat16 = self.resnet.layer3(feat8)
        feat32 = self.resnet.layer4(feat16)

        avg = torch.mean(feat32, dim=(2, 3), keepdim=True)    # [B, 512, 1, 1]
        avg = self.conv_avg(avg)                              # [B, 128, 1, 1]

        feat32_arm = self.arm32(feat32)                       # [B, 128, 15, 15]
        feat32_sum = feat32_arm + avg                         # [B, 128, 15, 15] 融合了feat32与center
        feat32_up = self.up32(feat32_sum)                     # [B, 128, 30, 30]
        feat32_up = self.conv_head32(feat32_up)               # [B, 128, 30, 30]

        feat16_arm = self.arm16(feat16)                       # [B, 128, 30, 30]
        feat16_sum = feat16_arm + feat32_up                   # [B, 128, 30, 30] 融合了feat16与feat32
        feat16_up = self.up16(feat16_sum)                     # [B, 128, 60, 60]
        feat16_up = self.conv_head16(feat16_up)               # [B, 128, 60, 60]

        return feat16_up, feat32_up # x8, x16

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)

    def get_params(self):
        wd_params, nowd_params = [], []
        for name, module in self.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                wd_params.append(module.weight)
                if not module.bias is None:
                    nowd_params.append(module.bias)
            elif isinstance(module, nn.modules.batchnorm._BatchNorm):
                nowd_params += list(module.parameters())
        return wd_params, nowd_params


class SpatialPath(nn.Module):
    def __init__(self, *args, **kwargs):
        super(SpatialPath, self).__init__()
        self.conv1 = ConvBNReLU(3, 64, ks=7, stride=2, padding=3)
        self.conv2 = ConvBNReLU(64, 64, ks=3, stride=2, padding=1)
        self.conv3 = ConvBNReLU(64, 64, ks=3, stride=2, padding=1)

        self.d_conv1 = ConvBNReLU(1, 64, ks=7, stride=2, padding=3)
        self.d_conv2 = ConvBNReLU(64, 64, ks=3, stride=2, padding=1)
        self.d_conv3 = ConvBNReLU(64, 64, ks=3, stride=2, padding=1)

        self.conv_out = ConvBNReLU(64, 128, ks=1, stride=1, padding=0)
        self.init_weight()

    def forward(self, x, d):
        x1 = self.conv1(x)
        d1 = self.d_conv1(d)
        x1 += d1
        x2 = self.conv2(x1)
        d2 = self.d_conv2(d1)
        x2 += d2
        x3 = self.conv3(x2)
        d3 = self.d_conv3(d2)
        x3 += d3

        out = self.conv_out(x3)
        return out

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)

    def get_params(self):
        wd_params, nowd_params = [], []
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear) or isinstance(module, nn.Conv2d):
                wd_params.append(module.weight)
                if not module.bias is None:
                    nowd_params.append(module.bias)
            elif isinstance(module, nn.modules.batchnorm._BatchNorm):
                nowd_params += list(module.parameters())
        return wd_params, nowd_params


class FeatureFusionModule(nn.Module):
    def __init__(self, in_chan, out_chan, *args, **kwargs):
        super(FeatureFusionModule, self).__init__()
        self.convblk = ConvBNReLU(in_chan, out_chan, ks=1, stride=1, padding=0)
        ## use conv-bn instead of 2 layer mlp, so that tensorrt 7.2.3.4 can work for fp16
        self.conv = nn.Conv2d(out_chan, out_chan, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn = nn.BatchNorm2d(out_chan)
        self.init_weight()

    def forward(self, fsp, fcp):
        fcat = torch.cat([fsp, fcp], dim=1)
        feat = self.convblk(fcat)
        atten = torch.mean(feat, dim=(2, 3), keepdim=True)
        atten = self.conv(atten)
        atten = self.bn(atten)
        atten = atten.sigmoid()
        feat_atten = torch.mul(feat, atten)
        feat_out = feat_atten + feat
        return feat_out

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)

    def get_params(self):
        wd_params, nowd_params = [], []
        for name, module in self.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                wd_params.append(module.weight)
                if not module.bias is None:
                    nowd_params.append(module.bias)
            elif isinstance(module, nn.modules.batchnorm._BatchNorm):
                nowd_params += list(module.parameters())
        return wd_params, nowd_params


class BiSeDNet(nn.Module):

    def __init__(self, n_classes, aux=None, root='./encoding/models/pretrain', *args, **kwargs):
        super(BiSeDNet, self).__init__()
        self.cp = ContextPath(root=root)
        self.sp = SpatialPath()
        self.ffm = FeatureFusionModule(256, 256)
        self.conv_out = BiSeNetOutput(256, 256, n_classes, up_factor=8)
        self.aux = aux
        if self.aux is not None:
            self.conv_out16 = BiSeNetOutput(128, 64, n_classes, up_factor=8)
            self.conv_out32 = BiSeNetOutput(128, 64, n_classes, up_factor=16)
        self.init_weight()

    def forward(self, x, d):
        H, W = x.size()[2:]
        feat_cp8, feat_cp16 = self.cp(x)
        feat_sp = self.sp(x, d)
        feat_fuse = self.ffm(feat_sp, feat_cp8)

        feat_out = [self.conv_out(feat_fuse)]
        if self.aux is not None:
            feat_out16 = self.conv_out16(feat_cp8)
            feat_out32 = self.conv_out32(feat_cp16)
            feat_out += [feat_out16, feat_out32]
        return feat_out

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)

    def get_params(self):
        wd_params, nowd_params, lr_mul_wd_params, lr_mul_nowd_params = [], [], [], []
        for name, child in self.named_children():
            child_wd_params, child_nowd_params = child.get_params()
            if isinstance(child, (FeatureFusionModule, BiSeNetOutput)):
                lr_mul_wd_params += child_wd_params
                lr_mul_nowd_params += child_nowd_params
            else:
                wd_params += child_wd_params
                nowd_params += child_nowd_params
        return wd_params, nowd_params, lr_mul_wd_params, lr_mul_nowd_params


def get_bised(dataset='nyud', backbone='resnet18', pretrained=True, root='./encoding/models/pretrain',
               fuse_type='1stage', mrf_fuse_type='1stage', mmfs=None, mrfs=None, aux=None, **kwargs):
    from ...datasets import datasets
    model = BiSeDNet(datasets[dataset.lower()].NUM_CLASS, root=root, aux=aux)
    return model