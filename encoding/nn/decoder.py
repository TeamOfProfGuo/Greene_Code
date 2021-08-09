# encoding:utf-8

import torch
import torch.nn as nn
from torch.nn import functional as F
from ..utils import parse_setting
from .basic import BasicBlock, FCNHead
from .fuse import Fuse_Block

__all__ = ['Decoder']

class Decoder(nn.Module):
    def __init__(self, decode_feat, n_classes, dtype='base', aux=None, auxl=None, feat='m', **kwargs):
        super().__init__()
        self.aux = [int(e) for e in list(aux)] if aux is not None else None
        self.auxl = auxl
        decoder_dict = {'base': Base_Decoder, 'refine': None, }
        self.decoder = decoder_dict[dtype](decode_feat, n_classes, aux=aux, auxl=auxl, feat=feat, **kwargs)

    def forward(self, feats):
        return self.decoder(feats)


class Base_Decoder(nn.Module):
    def __init__(self, decode_feat, n_classes, fuse_type='1stage', mrfs=None, aux=None, auxl=None, feat='l'):
        super().__init__()

        self.feat, self.aux, self.auxl = feat, aux, auxl
        self.mrf_args = parse_setting(mrfs)
        mrf_att = self.mrf_args.pop('mrf', None)

        # decode_feat = [None, 64, 128, 256, 512]
        self.up4 = nn.Sequential(BasicBlock(decode_feat[4], decode_feat[4]), BasicBlock(decode_feat[4], decode_feat[3], upsample=True))
        self.up3 = nn.Sequential(BasicBlock(decode_feat[3], decode_feat[3]), BasicBlock(decode_feat[3], decode_feat[2], upsample=True))
        self.up2 = nn.Sequential(BasicBlock(decode_feat[2], decode_feat[2]), BasicBlock(decode_feat[2], decode_feat[1], upsample=True))

        self.level_fuse3 = Fuse_Block(decode_feat[3], shape=(30, 30), mmf_att=mrf_att, fuse_type=fuse_type, **self.mrf_args)
        self.level_fuse2 = Fuse_Block(decode_feat[2], shape=(60, 60), mmf_att=mrf_att, fuse_type=fuse_type, **self.mrf_args)
        self.level_fuse1 = Fuse_Block(decode_feat[1], shape=(120, 120), mmf_att=mrf_att, fuse_type=fuse_type, **self.mrf_args)

        self.out_conv = nn.Sequential(BasicBlock(decode_feat[1], 128, upsample=True), BasicBlock(128, 128),
                                      nn.Conv2d(128, n_classes, kernel_size=1, stride=1, padding=0, bias=True))

        if self.aux is not None:
            for i in self.aux:
                j = i - 1 if self.auxl == 'u' else i
                self.add_module('auxlayer' + str(i), FCNHead(decode_feat[j], n_classes))

    def forward(self, feats):
        if self.feat == 'm':
            l1, l2, l3, l4 = feats.m1, feats.m2, feats.m3, feats.m4
        elif self.feat == 'l':
            l1, l2, l3, l4 = feats.l1, feats.l2, feats.l3, feats.l4

        y4u = self.up4(l4)  # [B, 256, h/16, w/16]
        y3, _ = self.level_fuse3(y4u, l3)

        y3u = self.up3(y3)  # [B, 128, h/8, w/8]
        y2, _ = self.level_fuse2(y3u, l2)  # [B, 128, h/8, w/8]

        y2u = self.up2(y2)  # [B, 64, h/4, w/4]
        y1, _ = self.level_fuse1(y2u, l1)  # [B, 64, h/4, w/4]

        out = self.out_conv(y1)
        outputs = [F.interpolate(out, feats.in_size, mode='bilinear', align_corners=True)]

        if self.aux is not None:
            yd = {3: y3, 2: y2, 1: y1} if self.auxl == None else {4: y4u, 3: y3u, 2: y2u, 1: y1}
            for i in self.aux:
                aux_out = self.__getattr__("auxlayer" + str(i))(yd[i])
                aux_out = F.interpolate(aux_out, feats.in_size, mode='bilinear', align_corners=True)
                outputs.append(aux_out)
        return outputs
