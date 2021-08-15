# encoding:utf-8 
import re
import torch
import torch.nn as nn
from torch.nn import Parameter
import torch.nn.functional as F
from .chaatt import AttGate1, AttGate2, AttGate2a, AttGate2b, AttGate2d, AttGate2e, PSK, AttGate3, AttGate3a, AttGate3b, AttGate4c, AttGate5c, AttGate6, AttGate9
from .chaatt import GCGF_Module
from .posatt import PosAtt0, PosAtt1, PosAtt2, PosAtt3, PosAtt3a, PosAtt3c, PosAtt4, PosAtt4a, PosAtt5, PosAtt6, PosAtt6a
from .posatt import PosAtt7, PosAtt7a, PosAtt7b, PosAtt7d, PosAtt9, PosAtt9a, CMPA1, CMPA1a, CMPA2, CMPA2a
from .att import ContextBlock, FPA
from .basic import *

Module_Dict={'CA0':AttGate1, 'CA1':AttGate1, 'CA2':AttGate2, 'CA2a':AttGate2a, 'CA2b':AttGate2b, 'CA2d': AttGate2d,
             'CA3':AttGate3, 'CA3a':AttGate3a, 'CA3b':AttGate3b, 'PSK': PSK, 'CA2e':AttGate2e,
             'CA4c':AttGate4c, 'CA5c':AttGate5c, 'CA6':AttGate6, 'CA9':AttGate9, 'PA0':PosAtt0, 'PA1': PosAtt1,
             'PA2':PosAtt2, 'PA3':PosAtt3, 'PA3a':PosAtt3a, 'PA3c':PosAtt3c, 'PA4':PosAtt4, 'PA4a':PosAtt4a, 'PA5': PosAtt5,
             'PA6': PosAtt6, 'PA6a': PosAtt6a, 'PA7': PosAtt7, 'PA7a': PosAtt7a, 'PA7b': PosAtt7b, 'PA7d': PosAtt7d,
             'CB': ContextBlock, 'PA9': PosAtt9, 'PA9a':PosAtt9a, 'CMPA1': CMPA1, 'CMPA1a': CMPA1a, 'CMPA2': CMPA2, 'CMPA2a': CMPA2a,
             'GF':GCGF_Module}

__all__ = ['Fuse_Block',]


class Fuse_Block(nn.Module):
    def __init__(self, in_ch, shape=None, mmf_att=None, **kwargs):
        super().__init__()
        self.mmf_att = mmf_att

        if mmf_att in ['CA0', 'CA4c', 'CA5c', 'CB', 'PA9', 'PA9a']:
            self.mode ='late'
        elif mmf_att in Module_Dict.keys():
            self.mode = 'early'
        elif mmf_att == None:
            self.mode = None

        if self.mode == 'late':
            self.rgb_att = Module_Dict[self.mmf_att](in_ch, shape, **kwargs)
            self.dep_att = Module_Dict[self.mmf_att](in_ch, shape, **kwargs)
        elif self.mode == 'early':
            self.att_module = Module_Dict[self.mmf_att](in_ch, shape, **kwargs)

    def forward(self, x, dep):
        d = dep

        if self.mode == 'late': 
            out = self.rgb_att(x) + self.dep_att(dep)
        elif self.mode == 'early':  
            out = self.att_module(x, dep)      # 'CA6'这里需要注意顺序，rgb在前面，dep在后面，对dep进行reweight
        elif self.mode == None:
            out = x+dep
        return out, d


def get_proc(proc, in_ch):
    if proc is None:
        return None
    proc_module = nn.Sequential()
    if 'c' in proc:
        proc_module.add_module('conv',nn.Conv2d(in_ch, in_ch, kernel_size=3, padding=1))
    if 'b' in proc:
        proc_module.add_module('bn',nn.BatchNorm2d(in_ch))
    if 'r' in proc:
        proc_module.add_module('relu',nn.ReLU(inplace=True))
    return proc_module

# 'GF':  'mrf=GF1|att=pdl|gcf=merge-gc|gca=None'
# gca is the param for PDL, leave it fixed

class Level_Fuse(nn.Module):
    def __init__(self, in_ch, shape=None, pre=(False, False), lfb='rbb[2->2]', mrf_att=None, **kwargs):
        super().__init__()
        self.mrf_att = mrf_att
        if self.mrf_att == 'GF1':
            self.mrf_att, pre = 'GF', (True, False)
        self.pre_flag = pre
        if self.mrf_att is not None:
            self.fuse = Module_Dict[self.mrf_att](in_ch, shape=shape, **kwargs)
        self.rfb0 = customized_module(lfb, in_ch) if self.pre_flag[0] else nn.Identity()
        self.rfb1 = customized_module(lfb, in_ch) if self.pre_flag[1] else nn.Identity()

    def forward(self, y, x):  # y 深层网路， x 浅层网络
        x = self.rfb0(x)  # Refine feats from backbone
        if self.mrf_att is not None:
            out = self.fuse(y, x)
        else:
            out = y+x
        return self.rfb1(out)