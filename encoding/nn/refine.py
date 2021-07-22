# encoding:utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F
from .unet_att import Attention_block

__all__ = ['ResidualConvUnit', 'MultiResolutionFusion', 'ChainedResidualPool',
           'RefineNetBlock', 'RefineAttBlock']


class ResidualConvUnit(nn.Module):
    def __init__(self, features):
        super().__init__()

        self.conv1 = nn.Conv2d(features, features, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv2 = nn.Conv2d(features, features, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.relu(x)
        out = self.conv1(out)
        out = self.relu(out)
        out = self.conv2(out)
        return out + x


class MultiResolutionFusion(nn.Module):
    def __init__(self, out_feats, shapes):
        # shapes: [(n_features1, scale1), (n_features2, scale2)  ]
        super().__init__()
        _, min_scale = min(shapes, key=lambda x: x[1])

        self.scale_factors = []
        for i, shape in enumerate(shapes):
            feat, scale = shape
            self.scale_factors.append(scale // min_scale)
            self.add_module("resolve{}".format(i),
                            nn.Conv2d(feat, out_feats, kernel_size=3, stride=1, padding=1, bias=False))

    def forward(self, *xs):

        output = self.resolve0(xs[0])
        if self.scale_factors[0] != 1:
            output = nn.functional.interpolate(output, scale_factor=self.scale_factors[0], mode='bilinear', align_corners=True)

        for i, x in enumerate(xs[1:], 1): # the value for i starts from 1
            current_out = self.__getattr__("resolve{}".format(i))(x)
            if self.scale_factors[i] != 1:
                current_out = nn.functional.interpolate(current_out, scale_factor=self.scale_factors[i], mode='bilinear', align_corners=True)
            output += current_out

        return output


class ChainedResidualPool(nn.Module):
    def __init__(self, feats):
        super().__init__()

        self.relu = nn.ReLU(inplace=True)
        for i in range(1, 4):
            self.add_module("block{}".format(i),
                            nn.Sequential(nn.MaxPool2d(kernel_size=5, stride=1, padding=2),
                                          nn.Conv2d(feats, feats, kernel_size=3, stride=1, padding=1, bias=False)))

    def forward(self, x):
        x = self.relu(x)
        path = x

        for i in range(1, 4):
            path = self.__getattr__("block{}".format(i))(path)
            x = x + path

        return x


class BaseRefineNetBlock(nn.Module):
    def __init__(self, features, residual_conv_unit, chained_residual_pool, shapes, with_CRP=False):
        super().__init__()
        self.with_CRP = with_CRP
        for i, shape in enumerate(shapes):  # [(n_features1, scale1), (n_features2, scale2)]
            feats = shape[0]
            self.add_module("rcu{}".format(i), nn.Sequential(residual_conv_unit(feats), residual_conv_unit(feats)))

        if self.with_CRP:
            self.crp = chained_residual_pool(features)
        self.output_conv = residual_conv_unit(features)

    def forward(self, *xs):
        rcu_xs = []

        for i, x in enumerate(xs):
            rcu_xs.append(self.__getattr__("rcu{}".format(i))(x))

        if self.mrf is not None:
            out = self.mrf(*rcu_xs)
        else:
            out = rcu_xs[0]

        if self.with_CRP:
            out = self.crp(out)
        return self.output_conv(out)


class RefineNetBlock(BaseRefineNetBlock):
    def __init__(self, features, shapes, with_CRP=False):
        super().__init__(features, ResidualConvUnit, ChainedResidualPool, shapes, with_CRP=with_CRP)
        self.mrf = MultiResolutionFusion(features, shapes) if len(shapes) != 1 else None


class RefineAttBlock(BaseRefineNetBlock):
    def __init__(self, features, shapes, att_type2=None, n_cbr=1, with_bn=False):
        super().__init__(features, ResidualConvUnit, ChainedResidualPool, shapes)
        self.mrf = MultiScaleAtt(features, shapes, att_type=att_type2, n_cbr=n_cbr, with_bn=with_bn) if len(shapes) != 1 else None


class MultiScaleAtt(nn.Module):
    def __init__(self, out_feats, shapes, att_type=None, n_cbr=1, with_bn=False):
        # shapes: [(n_ch1, scale1), (n_ch2, scale2)]  小feature map在前面， 对大feature map(浅层）进行attention
        super().__init__()
        self.att_type = att_type
        _, min_scale = min(shapes, key=lambda x: x[1])

        self.scale_factors = [shape[1]//min_scale for shape in shapes]

        self.resolve0 = nn.Conv2d(shapes[0][0], out_feats, kernel_size=3, stride=1, padding=1, bias=False)
        self.resolve1 = nn.Conv2d(shapes[1][0], out_feats, kernel_size=3, stride=1, padding=1, bias=False)
        if att_type == 'AT1':
            self.att_module = Attention_block(F_g=out_feats, F_l=out_feats, F_int=128)
        elif att_type == 'AT2':
            self.att_module = AttBlock2(in_ch=out_feats, r=4, n_cbr=n_cbr, with_bn=with_bn)
        elif att_type == 'AT3':
            self.att_module = AttBlock3(in_ch=out_feats, r=4, n_cbr=n_cbr, with_bn=with_bn)

    def forward(self, *xs):

        out0 = self.resolve0(xs[0])
        if self.scale_factors[0] != 1:
            out0 = nn.functional.interpolate(out0, scale_factor=self.scale_factors[0], mode='bilinear', align_corners=True)

        out1 = self.resolve1(xs[1])
        if self.scale_factors[1] != 1:
            out1 = nn.functional.interpolate(out1, scale_factor=self.scale_factors[1], mode='bilinear', align_corners=True)

        if self.att_type == None:
            output = out0 + out1
        elif self.att_type == 'AT1':
            output = self.att_module(g=out0, x=out1) + out0
        elif self.att_type == 'AT2' or self.att_type == 'AT3':
            output = self.att_module(g=out0, x=out1)

        return output


class AttBlock2(nn.Module):
    def __init__(self, in_ch, r=16, n_cbr=1, with_bn=False):
        super().__init__()
        self.with_bn = with_bn
        self.n_cbr = n_cbr
        out_ch = max( in_ch//r, 32)
        self.cbr = nn.Sequential(nn.Conv2d(in_ch*2, out_ch, kernel_size=1, stride=1, padding=0, bias=True),
                                 nn.BatchNorm2d(out_ch),
                                 nn.ReLU(inplace=True))
        if self.n_cbr ==2:
            self.cbr2 = nn.Sequential(nn.Conv2d(out_ch, out_ch, kernel_size=1, stride=1, padding=0, bias=True),
                                      nn.BatchNorm2d(out_ch),
                                      nn.ReLU(inplace=True))

        self.conv2 = nn.Conv2d(out_ch, 1, kernel_size=1, stride=1, padding=0, bias=True)
        self.sigmoid = nn.Sigmoid()

        if self.with_bn:
            self.bn2 = nn.BatchNorm2d(1)

    def forward(self, g, x):
        m = torch.cat((g,x), dim=1)    # [B, 2c, h, w]
        m = self.cbr(m)                # [B, out_c, h, w]
        if self.n_cbr==2:
            m = self.cbr2(m)           # [B, out_c, h, w]

        att = self.conv2(m)      # [B, 1, h, w]
        if self.with_bn:
            att = self.bn2(att)
        att = self.sigmoid(att)  # [B, 1, h, w]

        out = torch.mul(g, att) + torch.mul(x, 1-att)  # [B, c, h, w]
        return out


class AttBlock3(nn.Module):
    def __init__(self, in_ch, r=16, n_cbr=1, with_bn=False):
        super().__init__()
        self.with_bn, self.n_cbr = with_bn, n_cbr
        out_ch = max(in_ch//r, 32)

        self.point_conv = nn.Conv2d(in_ch*2, out_ch, kernel_size=1, stride=1, padding=0, bias=True)
        self.avg_pool = nn.AdaptiveAvgPool2d((1,1))
        self.gap_conv = nn.Conv2d(in_ch*2, out_ch, kernel_size=1, stride=1, bias=True)

        self.bn1 = nn.BatchNorm2d(out_ch)
        self.relu1 = nn.ReLU(inplace=True)

        if self.n_cbr ==2:
            self.cbr2 = nn.Sequential(nn.Conv2d(out_ch, out_ch, kernel_size=1, stride=1, padding=0, bias=True),
                                      nn.BatchNorm2d(out_ch),
                                      nn.ReLU(inplace=True))

        self.conv2 = nn.Conv2d(out_ch, 1, kernel_size=1, stride=1, padding=0, bias=True)
        self.sigmoid = nn.Sigmoid()

        if self.with_bn:
            self.bn2 = nn.BatchNorm2d(1)

    def forward(self, g, x):
        m = torch.cat((g,x), dim=1)    # [B, 2c, h, w]
        point_m = self.point_conv(m)   # [B, out_c, h, w]
        gap_m = self.avg_pool(m)       # [B, 2c, 1, 1]
        gap_m = self.gap_conv(gap_m)   # [B, out_c, 1, 1]

        out = point_m + gap_m            # [B, out_c, h, w]
        out = self.relu1(self.bn1(out))  # [B, out_c, h, w]

        if self.n_cbr == 2:
            out = self.cbr2(out)         # [B, out_c, h, w]

        att = self.conv2(out)            # [B, 1, h, w]
        if self.with_bn:
            att = self.bn2(att)
        att = self.sigmoid(att)  # [B, 1, h, w]

        out = torch.mul(g, att) + torch.mul(x, 1-att)  # [B, c, h, w]
        return out

