# encoding:utf-8

import os
import copy
import torch
import torch.nn as nn
from torch.nn import Parameter
from addict import Dict
import torch.nn.functional as F
from ...nn import Fuse_Block, FCNHead
from ...nn import Decoder
from ..backbone import get_resnet18
from ...utils import parse_setting

# RFUNet: Res Fuse U-Net
__all__ = ['ACNet', 'get_acnet']


class ACNet(nn.Module):
    def __init__(self, n_classes=21, backbone='resnet18', pretrained=True, dilation=1, root='./encoding/models/pretrain', aux=None,
                 fuse_type='1stage', mrf_fuse_type='1stage', mmfs=None, mrfs=None, auxl=None, dtype='base', param=None, **kwargs):
        """ axu: '321', '32', '21', '3', '2', '1' """
        super(ACNet, self).__init__()
        self.param = param
        self.mmf_args = parse_setting(mmfs, sep_out='|', sep_in='=')
        mmf_att = self.mmf_args.pop('mmf', None)
        print('++++++mmf:{}, mmf_args:{}+++++++'.format(mmf_att, self.mmf_args))

        self.base = get_resnet18(input_dim=3, dilation=dilation, f_path=os.path.join(root, 'resnet18-5c106cde.pth'))

        self.layer0 = nn.Sequential(self.base.conv1, self.base.bn1, self.base.relu)  # [B, 64, h/2, w/2]
        self.layer1 = nn.Sequential(self.base.maxpool, self.base.layer1) # [B, 64, h/4, w/4]
        self.layer2 = self.base.layer2  # [B, 128, h/8, w/8]
        self.layer3 = self.base.layer3  # [B, 256, h/16, w/16]
        self.layer4 = self.base.layer4  # [B, 512, h/32, w/32]

        self.d_layer0 = nn.Sequential(nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False),
                                      copy.deepcopy(self.base.bn1),
                                      copy.deepcopy(self.base.relu))
        self.d_layer1 = copy.deepcopy(self.layer1)
        self.d_layer2 = copy.deepcopy(self.layer2)
        self.d_layer3 = copy.deepcopy(self.layer3)
        self.d_layer4 = copy.deepcopy(self.layer4)

        self.m_layer1 = copy.deepcopy(self.layer1)
        self.m_layer2 = copy.deepcopy(self.layer2)
        self.m_layer3 = copy.deepcopy(self.layer3)
        self.m_layer4 = copy.deepcopy(self.layer4)

        self.fuse0 = Fuse_Block(64, shape=(240, 240), mmf_att=mmf_att, fuse_type=fuse_type, **self.mmf_args)
        self.fuse1 = Fuse_Block(64, shape=(120, 120), mmf_att=mmf_att, fuse_type=fuse_type, **self.mmf_args)
        self.fuse2 = Fuse_Block(128, shape=(60, 60), mmf_att=mmf_att, fuse_type=fuse_type, **self.mmf_args)
        self.fuse3 = Fuse_Block(256, shape=(30, 30), mmf_att=mmf_att, fuse_type=fuse_type, **self.mmf_args)
        self.fuse4 = Fuse_Block(512, shape=(15, 15), mmf_att=mmf_att, fuse_type=fuse_type, **self.mmf_args)

        if self.param == '1':
            self.param1 = nn.Parameter(torch.tensor(1.0))
            self.param2 = nn.Parameter(torch.tensor(1.0))
            self.param3 = nn.Parameter(torch.tensor(1.0))
            self.param4 = nn.Parameter(torch.tensor(1.0))

        decode_feat = [64, 64, 128, 256, 512]
        d_args = {'dtype': 'base', 'aux': aux, 'auxl': auxl, 'feat': 'm', 'fuse_type': mrf_fuse_type, 'mrfs':mrfs}
        print('decoder setting {}'.format(d_args))
        self.decoder = Decoder(decode_feat, n_classes, **d_args)

    def forward(self, x, d):
        _, _, h, w = x.size()
        d0 = self.d_layer0(d)       # [B, 64, h/2, w/2]
        l0 = self.layer0(x)         # [B, 64, h/2, w/2]
        m0, _ = self.fuse0(l0, d0)  # [B, 64, h/2, w/2]

        d1 = self.d_layer1(d0)      # [B, 64, h/4, w/4]
        l1 = self.layer1(l0)        # [B, 64, h/4, w/4]
        m1 = self.m_layer1(m0)
        o1, _ = self.fuse1(l1, d1)  # [B, 64, h/4, w/4]
        m1 = m1 + o1 if self.param is None else m1 + self.param1*o1

        d2 = self.d_layer2(d1)      # [B, 128, h/8, w/8]
        l2 = self.layer2(l1)        # [B, 128, h/8, w/8]
        m2 = self.m_layer2(m1)
        o2, _ = self.fuse2(l2, d2)  # [B, 128, h/8, w/8]
        m2 = m2 + o2 if self.param is None else m2 + self.param2*o2

        d3 = self.d_layer3(d2)      # [B, 256, h/16, w/16]
        l3 = self.layer3(l2)        # [B, 256, h/16, w/16]
        m3 = self.m_layer3(m2)
        o3, _ = self.fuse3(l3, d3)  # [B, 256, h/16, w/16]
        m3 = m3 + o3 if self.param is None else m3 + self.param3*o3

        d4 = self.d_layer4(d3)      # [B, 512, h/32, w/32]
        l4 = self.layer4(l3)        # [B, 512, h/32, w/32]
        m4 = self.m_layer4(m3)
        o4, _ = self.fuse4(l4, d4)  # [B, 512, h/32, w/32]
        m4 = m4 + o4 if self.param is None else m4 + self.param4*o4

        feats = Dict({'m1': m1, 'm2': m2, 'm3': m3, 'm4': m4, 'in_size':(h,w)})
        # ======== decoder ========
        outputs = self.decoder(feats)
        return outputs


def get_acnet(dataset='nyud', backbone='resnet18', pretrained=True, dilation=1, root='./encoding/models/pretrain',
               fuse_type='1stage', mrf_fuse_type='1stage', mmfs=None, mrfs=None, auxl=None, dtype='base', param=None, **kwargs):
    from ...datasets import datasets
    model = ACNet(datasets[dataset.lower()].NUM_CLASS, backbone, pretrained, dilation=dilation, root=root,
                   fuse_type=fuse_type, mrf_fuse_type=mrf_fuse_type, mmfs=mmfs, mrfs=mrfs, auxl=auxl, dtype=dtype, param=param, **kwargs)
    return model
