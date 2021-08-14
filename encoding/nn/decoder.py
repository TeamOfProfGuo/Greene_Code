# encoding:utf-8

import torch
import torch.nn as nn
from torch.nn import functional as F
from ..utils import parse_setting
from .basic import BasicBlock, FCNHead, IRB_Block, LearnedUpUnit
from .fuse import Fuse_Block, Level_Fuse
from .apnb import *
from .afnb import *

__all__ = ['Decoder']

class Decoder(nn.Module):
    def __init__(self, n_classes, dtype='base', aux=None, auxl=None, feat='m', **kwargs):
        super().__init__()
        self.aux = [int(e) for e in list(aux)] if aux is not None else None
        self.auxl = auxl
        decoder_dict = {'base': Base_Decoder, 'irb': IRB_Decoder, }
        self.decoder = decoder_dict[dtype](n_classes, aux=aux, auxl=auxl, feat=feat, **kwargs)

    def forward(self, feats):
        return self.decoder(feats)


class Res_Up_Block(nn.Module):
    def __init__(self, in_ch, out_ch, upsample=True):
        super().__init__()
        self.bk1 = BasicBlock(in_ch, in_ch)
        self.bk2 = BasicBlock(in_ch, out_ch, upsample=upsample)

    def forward(self, x):
        feat1 = self.bk1(x)
        out = self.bk2(feat1)
        return out, feat1


class IRB_Up_Block(nn.Module):
    def __init__(self, in_feats):
        super().__init__()

        self.conv_unit = nn.Sequential(
            IRB_Block(2*in_feats, 2*in_feats),
            IRB_Block(2*in_feats, 2*in_feats),
            IRB_Block(2*in_feats, in_feats)
        )
        self.up_unit = LearnedUpUnit(in_feats)

    def forward(self, x):
        feats = self.conv_unit(x)
        return self.up_unit(feats), feats


class IRB_Decoder(nn.Module):
    def __init__(self, n_classes, feat='l', mrfs = None, aux=None, auxl = None, dan=None):
        super().__init__()

        self.aux = aux
        self.feat = feat
        self.dan = dan
        self.mrf_args = parse_setting(mrfs)
        mrf_att = self.mrf_args.pop('mrf', None)

        #decoder_feats = [256, 128, 64]
        decode_feat = [None, 64, 128, 256, 512]
        shapes = [None, (120, 120), (60, 60), (30, 30), (15, 15)]

        # Upsample Blocks
        self.up4 = IRB_Up_Block(int(decode_feat[4]/2))
        self.up3 = IRB_Up_Block(int(decode_feat[3]/2))
        self.up2 = IRB_Up_Block(int(decode_feat[2]/2))

        # Refine Blocks
        self.refine3 = Level_Fuse(decode_feat[3], shape=shapes[3], mrf_att=mrf_att,  **self.mrf_args)
        self.refine2 = Level_Fuse(decode_feat[2], shape=shapes[2], mrf_att=mrf_att,  **self.mrf_args)
        self.refine1 = Level_Fuse(decode_feat[1], shape=shapes[1], mrf_att=mrf_att,  **self.mrf_args)

        # Aux loss
        if self.aux:
            for i in self.aux:    # note 经过 self.up4 block 之后， 通道数减半
                self.add_module('aux'+str(i),
                                nn.Conv2d(decode_feat[int(i)]//2, n_classes, kernel_size=1, stride=1, padding=0, bias=True))

        self.out_conv = nn.Sequential(
            nn.Conv2d(decode_feat[1], n_classes, kernel_size=1, stride=1, padding=0, bias=True),
            LearnedUpUnit(n_classes),
            LearnedUpUnit(n_classes)
        )

        if self.dan in ['21af']:
            afn_args = dict(low_in_channels=decode_feat[2], high_in_channels=decode_feat[1], out_channels=decode_feat[1],
                            key_channels=decode_feat[1], value_channels=decode_feat[1], dropout=0.05, sizes=([1]), psp_size=(1, 3, 6, 8))
            print('afn_args 21 {}'.format(afn_args))
            self.fuse21 = AFNB(**afn_args)

    def forward(self, in_feats):
        f1 = in_feats['%s1' % self.feat]
        f2 = in_feats['%s2' % self.feat]
        f3 = in_feats['%s3' % self.feat]
        f4 = in_feats['%s4' % self.feat]

        y4u, y4a = self.up4(f4)         # [B, 256, 30, 30],  [B, 256, 15, 15]
        y3= self.refine3(y4u, f3)   # [B, 256, 30, 30]
        y3u, y3a = self.up3(y3)         # [B, 128, 60, 60],  [B, 128, 30, 30]
        y2 = self.refine2(y3u, f2)
        y2u, y2a = self.up2(y2)         # [B, 64, 120, 120], [B, 64, 60, 60]
        y1 = self.refine1(y2u, f1)   # [B, 64, 120, 120]
        if self.dan == '21af':
            y1 = self.fuse21(y2, y1)

        out_feats = [self.out_conv(y1)]

        if self.aux:
            for i in self.aux:
                yd = {4: y4a, 3: y3a, 2:y2a}
                aux_out = self.__getattr__("aux" + str(i))(yd[int(i)])
                out_feats.append(aux_out)
        return out_feats


class Base_Decoder(nn.Module):
    def __init__(self, n_classes, fuse_type='1stage', mrfs=None, aux=None, auxl=None, feat='l', dan=None, decode_feat=[None, 64, 128, 256, 512]):
        super().__init__()

        self.feat, self.aux, self.auxl, self.dan = feat, aux, auxl, dan
        self.mrf_args = parse_setting(mrfs)
        mrf_att = self.mrf_args.pop('mrf', None)

        # decode_feat = [None, 64, 128, 256, 512]
        self.up4 = Res_Up_Block(decode_feat[4], decode_feat[3])
        self.up3 = Res_Up_Block(decode_feat[3], decode_feat[2])
        self.up2 = Res_Up_Block(decode_feat[2], decode_feat[1])

        self.level_fuse3 = Fuse_Block(decode_feat[3], shape=(30, 30), mmf_att=mrf_att, fuse_type=fuse_type, **self.mrf_args)
        self.level_fuse2 = Fuse_Block(decode_feat[2], shape=(60, 60), mmf_att=mrf_att, fuse_type=fuse_type, **self.mrf_args)
        self.level_fuse1 = Fuse_Block(decode_feat[1], shape=(120, 120), mmf_att=mrf_att, fuse_type=fuse_type, **self.mrf_args)

        encode_feat = [None, 64, 128, 256, 512]
        for i in range(1, 4):
            if decode_feat[i] != encode_feat[i]:
                self.add_module('proc'+str(i), nn.Conv2d(encode_feat[i], decode_feat[i], kernel_size=1, stride=1))
            else:
                self.add_module('proc'+str(i), nn.Identity())

        self.out_conv = nn.Sequential(BasicBlock(decode_feat[1], 128, upsample=True), BasicBlock(128, 128),
                                      nn.Conv2d(128, n_classes, kernel_size=1, stride=1, padding=0, bias=True))

        if self.dan in ['21af', 'aaf']:
            afn_args = dict(low_in_channels=decode_feat[2], high_in_channels=decode_feat[1], out_channels=decode_feat[1],
                            key_channels=decode_feat[1], value_channels=decode_feat[1], dropout=0.05, sizes=([1]), psp_size = (1, 3, 6, 8))
            print('afn_args 21 {}'.format(afn_args))
            self.fuse21 = AFNB(**afn_args)
            if self.dan == 'aaf':
                afn_args = dict(low_in_channels=decode_feat[3], high_in_channels=decode_feat[2], out_channels=decode_feat[2],
                                key_channels=decode_feat[2], value_channels=decode_feat[2], dropout=0.05, sizes=([1]), psp_size=(1, 3, 6, 8))
                print('afn_args 32 {}'.format(afn_args))
                self.fuse32 = AFNB(**afn_args)

        elif self.dan == '321af':
            afn_args = dict(low_in_channels=decode_feat[2:4], high_in_channels=decode_feat[1], out_channels=decode_feat[1],
                            key_channels=decode_feat[1], value_channels=decode_feat[1], dropout=0.05, norm_type=None, psp_size=(1,3,6,8))
            self.fuse321= AFNM(**afn_args)

        if self.aux is not None:
            for i in self.aux:
                j = int(i)
                self.add_module('auxlayer' + str(j), FCNHead(decode_feat[j], n_classes))

    def forward(self, feats):
        if self.feat == 'm':
            l1, l2, l3, l4 = feats.m1, feats.m2, feats.m3, feats.m4
        elif self.feat == 'l':
            l1, l2, l3, l4 = feats.l1, feats.l2, feats.l3, feats.l4

        y4u, y4a = self.up4(l4)  # [B, 256, h/16, w/16], [B, 512, h/32, w/32]
        y3, _ = self.level_fuse3(y4u, l3)  # [B, 256, h/16, w/16]

        y3u, y3a = self.up3(y3)  # [B, 128, h/8, w/8]
        y2, _ = self.level_fuse2(y3u, l2)  # [B, 128, h/8, w/8]

        if self.dan == 'aaf':
            y2 = self.fuse32(y3, y2)

        y2u, y2a = self.up2(y2)  # [B, 64, h/4, w/4]
        y1, _ = self.level_fuse1(y2u, l1)  # [B, 64, h/4, w/4]

        if self.dan in ['21af', 'aaf']:
            y1 = self.fuse21(y2, y1)
        if self.dan == '321af':
            y1 = self.fuse321([y2, y3], y1)

        out = self.out_conv(y1)
        outputs = [F.interpolate(out, feats.in_size, mode='bilinear', align_corners=True)]

        if self.aux is not None:
            if self.auxl is None or self.auxl =='f':
                yd = {3: y3, 2: y2, 1: y1}
            elif self.auxl == 'a':
                yd = {4: y4a, 3: y3a, 2: y2a}
            for i in self.aux:
                i = int(i)
                aux_out = self.__getattr__("auxlayer" + str(i))(yd[i])
                # aux_out = F.interpolate(aux_out, feats.in_size, mode='bilinear', align_corners=True)
                outputs.append(aux_out)
        return outputs

