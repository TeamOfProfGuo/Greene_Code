# encoding:utf-8 
import re
import torch
import torch.nn as nn
from torch.nn import Parameter
import torch.nn.functional as F
from .chaatt import AttGate1, AttGate2, AttGate2a, AttGate2b, AttGate3, AttGate3a, AttGate3b, AttGate4c, AttGate5c, AttGate6, AttGate9
from .posatt import PosAtt0, PosAtt1, PosAtt2, PosAtt3, PosAtt3a, PosAtt3c, PosAtt4, PosAtt4a, PosAtt5, PosAtt6, PosAtt6a
from .posatt import PosAtt7, PosAtt7a, PosAtt7b, PosAtt7d, PosAtt9, PosAtt9a, CMPA1, CMPA1a, CMPA2, CMPA2a
from .att import ContextBlock, FPA

Module_Dict={'CA0':AttGate1, 'CA1':AttGate1, 'CA2':AttGate2, 'CA2a':AttGate2a, 'CA2b':AttGate2b, 'CA3':AttGate3, 'CA3a':AttGate3a, 'CA3b':AttGate3b,
             'CA4c':AttGate4c, 'CA5c':AttGate5c, 'CA6':AttGate6, 'CA9':AttGate9, 'PA0':PosAtt0, 'PA1': PosAtt1,
             'PA2':PosAtt2, 'PA3':PosAtt3, 'PA3a':PosAtt3a, 'PA3c':PosAtt3c, 'PA4':PosAtt4, 'PA4a':PosAtt4a, 'PA5': PosAtt5,
             'PA6': PosAtt6, 'PA6a': PosAtt6a, 'PA7': PosAtt7, 'PA7a': PosAtt7a, 'PA7b': PosAtt7b, 'PA7d': PosAtt7d,
             'CB': ContextBlock, 'PA9': PosAtt9, 'PA9a':PosAtt9a, 'CMPA1': CMPA1, 'CMPA1a': CMPA1a, 'CMPA2': CMPA2, 'CMPA2a': CMPA2a}

__all__ = ['Fuse_Block', 'RGBDFuse', 'Fuse2Stage', 'GAU_Fuse', 'LevelFuse']


class Fuse_Block(nn.Module):
    def __init__(self, in_ch, shape=None, fuse_type='gau', mmf_att=None, **kwargs):
        super().__init__()
        fuse_dict = {'1stage': RGBDFuse, '2stage': Fuse2Stage, 'gau': GAU_Fuse,}
        self.fb = fuse_dict[fuse_type](in_ch, shape=shape, mmf_att=mmf_att, **kwargs)        
        
    def forward(self, rgb, dep):
        return self.fb(rgb, dep)  


class RGBDFuse(nn.Module):
    def __init__(self, in_ch, shape=None, mmf_att=None, **kwargs):
        super().__init__()
        self.mmf_att = mmf_att
        if mmf_att in ['CA0', 'CA4c', 'CA5c', 'CB', 'PA9', 'PA9a']:
            self.mode ='late'
        elif mmf_att in Module_Dict.keys():
            self.mode = 'early'
        elif mmf_att == None:
            self.mode = None
        #print('fusion model {}'.format(self.mode))

        if self.mode == 'late':
            self.rgb_att = Module_Dict[self.mmf_att](in_ch, shape, **kwargs)
            self.dep_att = Module_Dict[self.mmf_att](in_ch, shape, **kwargs)
        elif self.mode == 'early':
            self.att_module = Module_Dict[self.mmf_att](in_ch, shape, **kwargs)

    def forward(self, x, dep):
        if self.mode == 'late': 
            out = self.rgb_att(x) + self.dep_att(dep)
        elif self.mode == 'early':  
            out = self.att_module(x, dep)      # 'CA6'这里需要注意顺序，rgb在前面，dep在后面，对dep进行reweight
        elif self.mode == None:
            out = x+dep
        return out, dep


class GAU_Fuse(nn.Module):
    def __init__(self, in_ch, shape=None, mmf_att=None, att_feat='rgb', update_dep=False, param=False):
        super().__init__()
        self.att_module = AttGate6(in_ch, update_dep=True) 
        self.att_feat = att_feat
        self.update_dep = update_dep
        self.param = param
        if self.param:
            self.alpha = Parameter(torch.zeros(1))  

    def forward(self, x, d):  # x is rgb/深层特征， d为dep/浅层特征
        if self.att_feat == 'rgb':
            rgb_out, dep_out = self.att_module(x, d)
            if self.param:
                dep_out = d + self.alpha*dep_out
            if self.update_dep:
                return rgb_out, dep_out
            else:
                return rgb_out, d    # d维持不变

        elif self.att_feat == 'dep':
            dep_out, rgb_out = self.att_module(d, x)  # rgb被reweight, dep=dep+reweighted_dep
            if self.param:
                rgb_out = x + self.param*rgb_out
            if self.update_dep:
                return rgb_out, dep_out
            else:
                return rgb_out, d


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

def parse_att(mmf_att):
    if '_' in mmf_att or '+' in mmf_att or '^' in mmf_att:
        module_list = re.split('\_|\+', mmf_att)
    if '_' in mmf_att:
        ops = '_'
    elif '+' in mmf_att:
        ops = '+'
    elif '^' in mmf_att:
        ops = '^'
    return module_list, ops


class Fuse2Stage(nn.Module):
    def __init__(self, in_ch, shape=None, mmf_att=None, mode='early', proc=None, param=None, **kwargs):
        super().__init__()
        self.mmf_att = mmf_att
        self.proc = proc
        module_list, self.ops = parse_att(mmf_att)

        if self.ops == '_':
            self.mode = mode
            if mode == 'late':
                self.module0_rgb = Module_Dict[module_list[0]](in_ch, shape)
                self.module0_dep = Module_Dict[module_list[0]](in_ch, shape)
                self.proc_module_rgb = get_proc(proc, in_ch)
                self.proc_module_dep = get_proc(proc, in_ch)
                self.module1 = RGBDFuse(in_ch=in_ch, mmf_att=module_list[1], shape=shape, **kwargs)
            elif mode == 'early':
                self.module0 = RGBDFuse(in_ch=in_ch, mmf_att=module_list[0], shape=shape, **kwargs)
                self.proc_module = get_proc(proc, in_ch)
                self.module1 = Module_Dict[module_list[1]](in_ch, shape)
                self.alpha = Parameter(torch.zeros(1)) if param else None

        elif self.ops in ['+', '^']:
            for i, module_name in enumerate(module_list):
                self.add_module('module{}'.format(i), RGBDFuse(in_ch=in_ch, mmf_att=module_name, shape=shape))

    def forward(self, x, d):
        if self.ops == '+':
            out = self.module0(x, d) + self.module1(x, d)
        elif self.ops == '^':
            out0 = self.module0(x, d)
            out1 = self.module1(x, d)
            out = torch.max(out0, out1)
        elif self.ops == '_':
            if self.mode == 'early':
                out0, _ = self.module0(x, d)
                if self.proc:
                    out0 = self.proc_module(out0)
                if self.alpha:
                    out = self.module1(out0)*self.alpha + out0
                else:
                    out = self.module1(out0)
            elif self.mode == 'late':
                out0_rgb, out0_dep = self.module0_rgb(x), self.module0_dep(d)
                if self.proc:
                    out0_rgb, out0_dep = self.proc_module_rgb(out0_rgb), self.proc_module_dep(out0_dep)
                out, _ = self.module1(out0_rgb, out0_dep)
        return out, d


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
