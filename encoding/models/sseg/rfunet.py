# encoding:utf-8

import os
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from ...nn import BasicBlock, AttGate1, AttGate2, AttGate3, AttGate3a, AttGate3b, AttGate4c, AttGate5c, AttGate6, AttGate9
from ...nn import PosAtt0, PosAtt1, PosAtt2, PosAtt3, PosAtt3a, PosAtt3c, PosAtt4, PosAtt4a, PosAtt5, PosAtt6, PosAtt6a
from ...nn import PosAtt7, PosAtt7a, PosAtt7b, PosAtt7d, PosAtt9, PosAtt9a, CMPA1, CMPA1a, CMPA2, CMPA2a
from ...nn import ContextBlock, FPA

# RFUNet: Res Fuse U-Net
__all__ = ['RFUNet', 'get_rfunet']


Module_Dict={'CA0':AttGate1, 'CA1':AttGate1, 'CA2':AttGate2, 'CA3':AttGate3, 'CA3a':AttGate3a, 'CA3b':AttGate3b,
                 'CA4c':AttGate4c, 'CA5c':AttGate5c, 'CA6':AttGate6, 'CA9':AttGate9, 'PA0':PosAtt0, 'PA1': PosAtt1,
                 'PA2':PosAtt2, 'PA3':PosAtt3, 'PA3a':PosAtt3a, 'PA3c':PosAtt3c, 'PA4':PosAtt4, 'PA4a':PosAtt4a, 'PA5': PosAtt5,
                 'PA6': PosAtt6, 'PA6a': PosAtt6a, 'PA7': PosAtt7, 'PA7a': PosAtt7a, 'PA7b': PosAtt7b, 'PA7d': PosAtt7d,
                 'CB': ContextBlock, 'PA9': PosAtt9, 'PA9a':PosAtt9a, 'CMPA1': CMPA1, 'CMPA1a': CMPA1a, 'CMPA2': CMPA2,
                 'CMPA2a': CMPA2a}


class RFUNet(nn.Module):
    def __init__(self, n_classes=21, backbone='resnet18', pretrained=True, root='./encoding/models/pretrain',
                 mmf_att=None, mrf_att=None):
        super(RFUNet, self).__init__()

        self.base = models.resnet18(pretrained=False)
        if pretrained:
            if backbone == 'resnet18':
                f_path = os.path.abspath(os.path.join(root, 'resnet18-5c106cde.pth'))
            if not os.path.exists(f_path):
                raise FileNotFoundError('the pretrained model can not be found')
            self.base.load_state_dict(torch.load(f_path), strict=False)

        self.dep_base = copy.deepcopy(self.base)
        self.dep_base.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)

        self.layer0 = nn.Sequential(self.base.conv1, self.base.bn1, self.base.relu)  # [B, 64, h/2, w/2]
        self.pool1 = self.base.maxpool  # [B, 64, h/4, w/4]
        self.layer1 = self.base.layer1  # [B, 64, h/4, w/4]
        self.layer2 = self.base.layer2  # [B, 128, h/8, w/8]
        self.layer3 = self.base.layer3  # [B, 256, h/16, w/16]
        self.layer4 = self.base.layer4  # [B, 512, h/32, w/32]

        self.d_layer0 = nn.Sequential(self.dep_base.conv1, self.dep_base.bn1, self.dep_base.relu)
        self.d_pool1 = self.dep_base.maxpool
        self.d_layer1 = self.dep_base.layer1
        self.d_layer2 = self.dep_base.layer2
        self.d_layer3 = self.dep_base.layer3
        self.d_layer4 = self.dep_base.layer4

        self.fuse0 = RGBDFuse(64, mmf_att=mmf_att, shape=(240, 240))
        self.fuse1 = RGBDFuse(64, mmf_att=mmf_att, shape=(120, 120))
        self.fuse2 = RGBDFuse(128,mmf_att=mmf_att, shape=(60, 60))
        self.fuse3 = RGBDFuse(256,mmf_att=mmf_att, shape=(30, 30))
        self.fuse4 = RGBDFuse(512,mmf_att=mmf_att, shape=(15, 15))

        self.up4 = nn.Sequential(BasicBlock(512, 512), BasicBlock(512, 256, upsample=True))
        self.up3 = nn.Sequential(BasicBlock(256, 256), BasicBlock(256, 128, upsample=True))
        self.up2 = nn.Sequential(BasicBlock(128, 128), BasicBlock(128, 64, upsample=True))

        self.level_fuse3 = LevelFuse(256, mrf_att=mrf_att)
        self.level_fuse2 = LevelFuse(128, mrf_att=mrf_att)
        self.level_fuse1 = LevelFuse(64, mrf_att=mrf_att)

        self.out_conv = nn.Sequential(BasicBlock(64, 128, upsample=True), BasicBlock(128, 128),
                                      nn.Conv2d(128, n_classes, kernel_size=1, stride=1, padding=0, bias=True))

    def forward(self, x, d):
        _, _, h, w = x.size()
        d0 = self.d_layer0(d)  # [B, 64, h/2, w/2]
        x0 = self.layer0(x)  # [B, 64, h/2, w/2]
        l0, d0 = self.fuse0(x0, d0)  # [B, 64, h/2, w/2]

        d1 = self.d_pool1(d0)  # [B, 64, h/4, w/4]
        d1 = self.d_layer1(d1)  # [B, 64, h/4, w/4]
        l1 = self.pool1(l0)  # [B, 64, h/4, w/4]
        l1 = self.layer1(l1)  # [B, 64, h/4, w/4]
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

        y4 = self.up4(l4)  # [B, 256, h/16, w/16]
        y3 = self.level_fuse3(y4, l3)

        y3 = self.up3(y3)  # [B, 128, h/8, w/8]
        y2 = self.level_fuse2(y3, l2)  # [B, 128, h/8, w/8]

        y2 = self.up2(y2)  # [B, 64, h/4, w/4]
        y1 = self.level_fuse1(y2, l1)  # [B, 64, h/4, w/4]

        out = self.out_conv(y1)
        out = F.interpolate(out, (h, w), mode='bilinear', align_corners=True)
        return out


def get_rfunet(dataset='nyud', backbone='resnet18', pretrained=True, root='./encoding/models/pretrain',
               mmf_att=None, mrf_att=None ):
    from ...datasets import datasets
    model = RFUNet(datasets[dataset.lower()].NUM_CLASS, backbone, pretrained, root=root,
                   mmf_att=mmf_att, mrf_att=mrf_att )
    return model


class RGBDFuse(nn.Module):
    def __init__(self, in_ch, mmf_att=None, shape=None, **kwargs):
        super().__init__()

        self.mmf_att = mmf_att 

        self.mode = 'late' if mmf_att in ['CA0', 'CA4c', 'CA5c', 'CB', 'PA9', 'PA9a'] else 'early' 
        print('fusion model {}'.format(self.mode))

        if self.mode == 'late':
            self.rgb_att = Module_Dict[self.mmf_att](in_ch, shape, **kwargs)
            self.dep_att = Module_Dict[self.mmf_att](in_ch, shape, **kwargs)
        else:
            self.att_module = Module_Dict[self.mmf_att](in_ch, shape, **kwargs)


    def forward(self, x, dep):
        batch_size, ch, _, _ = x.size()
        if self.mode == 'late': 
            out = self.rgb_att(x) + self.dep_att(dep)
        
        elif self.mode == 'early':  
            out = self.att_module(x, dep)      # 'CA6'这里需要注意顺序，rgb在前面，dep在后面，对dep进行reweight
        return out, dep


class LevelFuse(nn.Module):
    def __init__(self, in_ch, mrf_att=None):
        super().__init__()
        self.mrf_att = mrf_att
        if mrf_att == 'PA0':
            self.att_module = PosAtt0(ch=in_ch)
            self.out_conv = nn.Conv2d(in_ch * 2, in_ch, kernel_size=1, stride=1)
        elif mrf_att == 'CA6':
            self.att_module = AttGate6(in_ch=in_ch)

    def forward(self, c, x):
        if self.mrf_att in ['CA6']:
            return self.att_module(c, x) # 注意深层feature在前，浅层feature在后，对浅层feature进行变换
        elif self.mrf_att == 'PA0':
            x = self.att_module(c, x)
            return self.out_conv( torch.cat((c, x), dim=1) )
        else:
            return c + x
