# encoding:utf-8

import os
from copy import deepcopy
import torch
import torch.nn as nn
from addict import Dict
from ...utils import parse_setting
from ...nn import BasicBlock, Decoder, APNB
from ...nn import Fuse_Block, FCNHead
from ..backbone import get_backbone

# RFUNet: Res Fuse U-Net
__all__ = ['RFUNet', 'get_rfunet']


class RFUNet(nn.Module):
    def __init__(self, n_classes=21, backbone='resnet18', pretrained=True, dilation=1, root='./encoding/models/pretrain', aux=None,
                 fuse_type='1stage', refine=None, mmfs=None, mrfs=None, auxl='a', dtype='irb', ctr=None, dan=None, out=None, **kwargs):
        """ axu: '321', '32', '21', '3', '2', '1' """
        super(RFUNet, self).__init__()
        self.ctr, self.dtype = ctr, dtype
        self.mmf_args = parse_setting(mmfs, sep_out='|', sep_in='=')
        mmf_att = self.mmf_args.pop('mmf', None)
        print('++++++aux:{}++++++auxl:{}++++++mmf:{}, mmf_args:{}+++++++'.format(aux, auxl, mmf_att, self.mmf_args))

        self.base = get_backbone(backbone=backbone, input_dim=3, pretrained=True, root=root)

        if backbone in ['resnet18', 'resnet50']:
            self.layer0 = nn.Sequential(self.base.conv1, self.base.bn1, self.base.relu)  # [B, 64, h/2, w/2]
            self.d_layer0 = nn.Sequential(
                nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False),
                deepcopy(self.base.bn1),
                deepcopy(self.base.relu))
        else:
            self.layer0 = nn.Sequential(self.base.conv1, self.base.bn1, self.base.relu,
                                        self.base.conv2, self.base.bn2, self.base.relu,
                                        self.base.conv3, self.base.bn3, self.base.relu,)  # [B, 64, h/2, w/2]
            self.d_layer0= nn.Sequential(deepcopy(self.base.conv1), deepcopy(self.base.bn1), self.base.relu,
                                         deepcopy(self.base.conv2), deepcopy(self.base.bn2), self.base.relu,
                                         deepcopy(self.base.conv3), deepcopy(self.base.bn3), self.base.relu,)  # [B, 64, h/2, w/2]

        self.layer1 = nn.Sequential(self.base.maxpool, self.base.layer1) # [B, 64, h/4, w/4]
        self.layer2 = self.base.layer2  # [B, 128, h/8, w/8]
        self.layer3 = self.base.layer3  # [B, 256, h/16, w/16]
        self.layer4 = self.base.layer4  # [B, 512, h/32, w/32]

        self.d_layer1 = deepcopy(self.layer1)
        self.d_layer2 = deepcopy(self.layer2)
        self.d_layer3 = deepcopy(self.layer3)
        self.d_layer4 = deepcopy(self.layer4)

        feat_ch=[64, 64, 128, 256, 512] if backbone in ['resnet18','resnet34'] else [64, 256, 512, 1024, 2048]
        self.fuse0 = Fuse_Block(feat_ch[0], shape=(240, 240), mmf_att=mmf_att, **self.mmf_args)
        self.fuse1 = Fuse_Block(feat_ch[1], shape=(120, 120), mmf_att=mmf_att, **self.mmf_args)
        self.fuse2 = Fuse_Block(feat_ch[2], shape=(60, 60), mmf_att=mmf_att, **self.mmf_args)
        self.fuse3 = Fuse_Block(feat_ch[3], shape=(30, 30), mmf_att=mmf_att, **self.mmf_args)
        self.fuse4 = Fuse_Block(feat_ch[4], shape=(15, 15), mmf_att=mmf_att, **self.mmf_args)

        if self.ctr=='apn':
            apn_args = {'in_channels':feat_ch[4], 'out_channels':feat_ch[4], 'key_channels':256, 'value_channels':512, 'dropout':0.05,
                        'sizes':([1]), 'norm_type':'BN', 'psp_size':(1,3,5,7)}
            print('settings for APNB is {}'.format(apn_args))
            self.ctr_blk = APNB(**apn_args)

        d_args = {'dtype': dtype, 'aux': aux, 'auxl': auxl, 'feat': 'l', 'mrfs': mrfs, 'dan': dan, 'out':out,  # out:是否采用FCN Head
                  'decode_feat': feat_ch}
        print('decoder setting {}'.format(d_args))
        self.decoder = Decoder(n_classes, **d_args)

    def forward(self, x, d):
        _, _, h, w = x.size()
        d0 = self.d_layer0(d)  # [B, 64, h/2, w/2]
        x0 = self.layer0(x)  # [B, 64, h/2, w/2]
        l0, d0 = self.fuse0(x0, d0)  # [B, 64, h/2, w/2]

        d1 = self.d_layer1(d0)  # [B, 64, h/4, w/4]
        l1 = self.layer1(l0)  # [B, 64, h/4, w/4]
        l1, d1= self.fuse1(l1, d1)  # [B, 64, h/4, w/4]

        d2 = self.d_layer2(d1)  # [B, 128, h/8, w/8]
        l2 = self.layer2(l1)  # [B, 128, h/8, w/8]
        l2, d2 = self.fuse2(l2, d2)  # [B, 128, h/8, w/8] 

        d3 = self.d_layer3(d2)
        l3 = self.layer3(l2)  # [B, 256, h/16, w/16]
        l3, d3= self.fuse3(l3, d3)  # [B, 256, h/16, w/16]

        d4 = self.d_layer4(d3)
        l4 = self.layer4(l3)  # [B, 512, h/32, w/32]
        l4, _ = self.fuse4(l4, d4)  # [B, 512, h/32, w/32]

        if self.ctr=='apn':
            l4 = self.ctr_blk(l4)

        feats = Dict({'l1': l1, 'l2': l2, 'l3': l3, 'l4': l4, 'in_size': (h, w)})
        # ======== decoder ========
        outputs = self.decoder(feats)
        return outputs

def get_rfunet(dataset='nyud', backbone='resnet18', pretrained=True, dilation=1, root='./encoding/models/pretrain',
               fuse_type='1stage', mrf_fuse_type='1stage', mmfs=None, mrfs=None, auxl='a', **kwargs):
    from ...datasets import datasets
    model = RFUNet(datasets[dataset.lower()].NUM_CLASS, backbone, pretrained, dilation=dilation, root=root,
                   fuse_type=fuse_type, mmfs=mmfs, mrfs=mrfs, auxl=auxl, **kwargs)
    return model

