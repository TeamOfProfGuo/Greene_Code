#encoding:utf-8
import torch
import numpy as np
import torch.nn as nn
from functools import reduce
from torch.nn import Module, Softmax, Parameter
from .center import PyramidPooling
__all__ = ['AttGate1', 'AttGate2', 'AttGate2a', 'AttGate2b', 'AttGate2d', 'AttGate3', 'AttGate3a', 'AttGate3b', 'AttGate4c',
           'AttGate5c', 'AttGate6', 'AttGate9']


# class AttGate2(Module):
#     """ Channel attention module"""
#     def __init__(self, in_ch, reduce_rate=16):
#         super(AttGate2, self).__init__()
#         self.global_avg = nn.AdaptiveAvgPool2d(1)
#         fc_ch = max(in_ch//reduce_rate, 32)
#         self.fc = nn.Sequential(nn.Conv2d(in_ch, fc_ch, kernel_size=1, stride=1, bias=False),
#                                 nn.BatchNorm2d(num_features=fc_ch),
#                                 nn.ReLU(inplace=True))
#         self.a_linear = nn.Conv2d(fc_ch, in_ch, kernel_size=1, stride=1)
#         self.b_linear = nn.Conv2d(fc_ch, in_ch, kernel_size=1, stride=1)
#         self.softmax = Softmax(dim=2)
#
#     def forward(self, x, y):
#         """
#         inputs : x : input feature maps( B X C X H X W); y : input feature maps( B X C X H X W)
#         returns : out: [B, c, h, w]; attention [B, c, 1, 1] for both x and y
#         """
#         u = self.global_avg(x + y)                          # [B, c, 1, 1]
#         z = self.fc(u)                                      # [B, d, 1, 1]
#         a_att, b_att = self.a_linear(z), self.b_linear(z)   # [B, c, 1, 1]
#         att = torch.cat((a_att, b_att), dim=2)              # [B, c, 2, 1]
#         att = self.softmax(att)                             # [B, c, 2, 1]
#
#         out = torch.mul(x, att[:, :, 0:1, :]) + torch.mul(y, att[:, :, 1:2, :])
#         return out

class AttGate1(nn.Module):
    def __init__(self, in_ch, r=4):
        """same as the channel attention in SE module"""
        super(AttGate1, self).__init__()
        int_ch = max(in_ch//r, 32)
        self.gap = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Sequential(nn.Conv2d(in_ch, int_ch, kernel_size=1, stride=1),
                                nn.BatchNorm2d(int_ch),
                                nn.ReLU(inplace=True),
                                nn.Conv2d(int_ch, in_ch, kernel_size=1, stride=1),
                                nn.Sigmoid())

    def forward(self, x):
        att = self.gap(x)
        att = self.fc(att)  # [B, in_c, 1, 1]
        out = att*x
        return out


class AttGate2(nn.Module):
    def __init__(self, in_ch, shape=None, M=2, r=4, ret_att=False):
        """ Attention as in SKNet (selective kernel)
        Args:
            features/in_ch: input channel dimensionality.
            M: the number of branches.
            r: the ratio for compute d, the length of z.
            L: the minimum dim of the vector z in paper, default 32.
        """
        super(AttGate2, self).__init__()
        print('Att in_ch {} type {}, r {} type {}'.format(in_ch, type(in_ch), r, type(r)))
        d = max(int(in_ch / r), 32)
        self.M = M
        self.in_ch = in_ch
        self.ret_att = ret_att
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        # to calculate Z
        self.fc = nn.Sequential(nn.Conv2d(in_ch, d, kernel_size=1, stride=1, bias=False),
                                nn.BatchNorm2d(d),
                                nn.ReLU(inplace=True))
        # 各个分支
        self.fcs = nn.ModuleList([])
        for i in range(M):
            self.fcs.append(nn.Conv2d(d, in_ch, kernel_size=1, stride=1))
        self.softmax = nn.Softmax(dim=1)

    def forward(self, *inputs):

        U = reduce(lambda x, y: x+y, inputs)
        batch_size = U.size(0)

        S = self.gap(U)  # [B, c, 1, 1]
        Z = self.fc(S)   # [B, d, 1, 1]

        attention_vectors = [fc(Z) for fc in self.fcs]           # M: [B, c, 1, 1]
        attention_vectors = torch.cat(attention_vectors, dim=1)  # [B, Mc, 1, 1]
        attention_vectors = attention_vectors.view(batch_size, self.M, self.in_ch, 1, 1)  # [B, M, c, 1, 1]
        attention_vectors = self.softmax(attention_vectors)      # [B, M, c, 1, 1]

        feats = torch.cat(inputs, dim=1)  # [B, Mc, h, w]
        feats = feats.view(batch_size, self.M, self.in_ch, feats.shape[2], feats.shape[3])  # [B, M, c, h, w]
        feats_V = torch.sum(feats * attention_vectors, dim=1)

        if self.ret_att:
            return feats_V, attention_vectors
        else:
            return feats_V


class AttGate2a(nn.Module):
    def __init__(self, in_ch, shape=None, r=4, act_fn=None):
        """ Attention as in SKNet (selective kernel) """
        super().__init__()
        d = max(int(in_ch / r), 32)
        self.act_fn = act_fn
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        # to calculate Z
        self.fc = nn.Sequential(nn.Conv2d(in_ch, d, kernel_size=1, stride=1, bias=False),
                                nn.BatchNorm2d(d),
                                nn.ReLU(inplace=True))
        # 各个分支
        self.fc_x = nn.Conv2d(d, in_ch, kernel_size=1, stride=1)
        self.fc_y = nn.Conv2d(d, in_ch, kernel_size=1, stride=1)
        if act_fn == 'sigmoid':
            self.act_x = nn.Sigmoid()
            self.act_y = nn.Sigmoid()
        elif act_fn == 'tanh':
            self.act_x = nn.Sequential(nn.ReLU(inplace=True), nn.Tanh())
            self.act_y = nn.Sequential(nn.ReLU(inplace=True), nn.Tanh())
        elif act_fn == 'rsigmoid':
            self.act_x = nn.Sequential(nn.ReLU(inplace=True), nn.Sigmoid())
            self.act_y = nn.Sequential(nn.ReLU(inplace=True), nn.Sigmoid())
        elif act_fn == 'softmax':
            self.act = nn.Softmax(dim=1)

    def forward(self, x, y):
        U = x+y
        batch_size, ch, _, _ = U.size()

        S = self.gap(U)     # [B, c, 1, 1]
        Z = self.fc(S)      # [B, d, 1, 1]

        z_x = self.fc_x(Z)  # [B, c, 1, 1]
        z_y = self.fc_y(Z)  # [B, c, 1, 1]
        if self.act_fn in ['sigmoid', 'tanh', 'rsigmoid']:
            w_x = self.act_x(z_x)    # [B, c, 1, 1]
            w_y = self.act_y(z_y)    # [B, c, 1, 1]
        elif self.act_fn == 'softmax':
            w_xy = torch.cat((z_x, z_y), dim=1)        # [B, 2c, 1, 1]
            w_xy = w_xy.view(batch_size, 2, ch, 1, 1)  # [B, 2, c, 1, 1]
            w_xy = self.act(w_xy)                      # [B, 2, c, 1, 1]
            w_x, w_y = w_xy[:, 0].contiguous(), w_xy[:, 1].contiguous()      # [B, c, 1, 1]
        out = w_x*x + w_y*y
        return out


class AttGate2b(nn.Module):
    def __init__(self, in_ch, shape=None, r=16, act_fn=None):
        """ Attention as in SKNet (selective kernel) """
        super().__init__()
        d = max(int(in_ch / 16), 32)
        self.act_fn = act_fn
        self.pool1 = nn.AdaptiveAvgPool2d((1, 1))
        self.pool3 = nn.AdaptiveAvgPool2d((3, 3))
        # to calculate Z
        self.fc = nn.Sequential(nn.Linear(in_ch*10, d, bias=False),
                                nn.BatchNorm1d(d),
                                nn.ReLU(inplace=True))
        # 各个分支
        self.fc_x = nn.Linear(d, in_ch)
        self.fc_y = nn.Linear(d, in_ch)
        if act_fn == 'sigmoid':
            self.act_x = nn.Sigmoid()
            self.act_y = nn.Sigmoid()
        elif act_fn == 'tanh':
            self.act_x = nn.Sequential(nn.ReLU(inplace=True), nn.Tanh())
            self.act_y = nn.Sequential(nn.ReLU(inplace=True), nn.Tanh())
        elif act_fn == 'rsigmoid':
            self.act_x = nn.Sequential(nn.ReLU(inplace=True), nn.Sigmoid())
            self.act_y = nn.Sequential(nn.ReLU(inplace=True), nn.Sigmoid())
        elif act_fn == 'softmax':
            self.act = nn.Softmax(dim=1)

    def forward(self, x, y):
        U = x+y
        batch_size, ch, _, _ = U.size()

        s1 = self.pool1(U).view(batch_size, ch, -1)    # [B, c, 1]
        s2 = self.pool3(U).view(batch_size, ch, -1)    # [B, c, 9]
        s = torch.cat((s1,s2), dim=2).view(batch_size, -1).contiguous()  # [B, 10*c]
        z = self.fc(s)      # [B, d]

        z_x = self.fc_x(z)  # [B, c]
        z_y = self.fc_y(z)  # [B, c]
        if self.act_fn in ['sigmoid', 'tanh', 'rsigmoid']:
            w_x = self.act_x(z_x)    # [B, c]
            w_y = self.act_y(z_y)    # [B, c]
        elif self.act_fn == 'softmax':
            w_xy = torch.cat((z_x, z_y), dim=1)    # [B, 2c]
            w_xy = w_xy.view(batch_size, 2, ch)    # [B, 2, c]
            w_xy = self.act(w_xy)                  # [B, 2, c]
            w_x, w_y = w_xy[:, 0].contiguous(), w_xy[:, 1].contiguous()      # [B, c]
        out = x * w_x.view(batch_size, ch, 1, 1) + y * w_y.view(batch_size, ch, 1, 1)
        return out


class AttGate2d(nn.Module):
    def __init__(self, in_ch, shape=None, r=16, thold=15, act_fn=None):
        """ Attention as in SKNet (selective kernel) """
        super().__init__()
        d = max(int(in_ch / r), 32)
        self.act_fn = act_fn
        self.pool1 = nn.AdaptiveAvgPool2d((1, 1))
        self.pool3 = nn.AdaptiveAvgPool2d((3, 3))
        self.pool5 = nn.AdaptiveAvgPool2d((5, 5))
        self.pool10 = nn.AdaptiveAvgPool2d((10, 10)) if shape[0]>thold else None
        # dimension reduction
        dr_in_ch = 135 if self.pool10 else 35
        dr_out_ch = 8
        self.dr = nn.Conv1d(dr_in_ch, dr_out_ch, kernel_size=1, stride=1)

        # to calculate Z
        self.fc = nn.Sequential(nn.Linear(in_ch*dr_out_ch, d, bias=False),
                                nn.BatchNorm1d(d),
                                nn.ReLU(inplace=True))
        # 各个分支
        self.fc_x = nn.Linear(d, in_ch)
        self.fc_y = nn.Linear(d, in_ch)
        if act_fn == 'sigmoid':
            self.act_x = nn.Sigmoid()
            self.act_y = nn.Sigmoid()
        elif act_fn == 'tanh':
            self.act_x = nn.Sequential(nn.ReLU(inplace=True), nn.Tanh())
            self.act_y = nn.Sequential(nn.ReLU(inplace=True), nn.Tanh())
        elif act_fn == 'rsigmoid':
            self.act_x = nn.Sequential(nn.ReLU(inplace=True), nn.Sigmoid())
            self.act_y = nn.Sequential(nn.ReLU(inplace=True), nn.Sigmoid())
        elif act_fn == 'softmax':
            self.act = nn.Softmax(dim=1)

    def forward(self, x, y):
        U = x+y
        batch_size, ch, _, _ = U.size()

        s1 = self.pool1(U).view(batch_size, ch, -1)    # [B, c, 1]
        s3 = self.pool3(U).view(batch_size, ch, -1)    # [B, c, 9]
        s5 = self.pool5(U).view(batch_size, ch, -1)    # [B, c, 25]
        s10 = self.pool10(U).view(batch_size, ch, -1) if self.pool10 else None  # [B, c, 100]
        s = s1
        for feat in [s3, s5, s10]:
            if feat is not None:
                s = torch.cat((s, feat), dim=2)           # [B, c, 1+9+25+100]
        s = s.permute(0, 2, 1).contiguous()               # [B, 135, c]
        s = self.dr(s).view(batch_size, -1).contiguous()  # [B, 8, c] -> [B, 8*c]

        z = self.fc(s)      # [B, d]

        z_x = self.fc_x(z)  # [B, c]
        z_y = self.fc_y(z)  # [B, c]
        if self.act_fn in ['sigmoid', 'tanh', 'rsigmoid']:
            w_x = self.act_x(z_x)    # [B, c]
            w_y = self.act_y(z_y)    # [B, c]
        elif self.act_fn == 'softmax':
            w_xy = torch.cat((z_x, z_y), dim=1)    # [B, 2c]
            w_xy = w_xy.view(batch_size, 2, ch)    # [B, 2, c]
            w_xy = self.act(w_xy)                  # [B, 2, c]
            w_x, w_y = w_xy[:, 0].contiguous(), w_xy[:, 1].contiguous()      # [B, c]
        out = x * w_x.view(batch_size, ch, 1, 1) + y * w_y.view(batch_size, ch, 1, 1)
        return out


class AttGate3(nn.Module):
    def __init__(self, in_ch, shape=None, M=2, r=4):
        # 输入特征的通道数， 2个分支，bottle-net layer的 reduction rate
        super(AttGate3, self).__init__()
        d = max(int(in_ch*2 / r), 32)
        self.M = M
        self.in_ch = in_ch
        self.gap = nn.AdaptiveAvgPool2d((1, 1))

        # to calculate Z
        self.fc1 = nn.Sequential(nn.Conv2d(in_ch*2, d, kernel_size=1, stride=1, bias=False),
                                nn.BatchNorm2d(d),
                                nn.ReLU(inplace=True))
        self.fc2 = nn.Conv2d(d, in_ch*2, kernel_size=1, stride=1, bias=False)

        # to calculate attention score
        self.softmax = nn.Softmax(dim=1)

    def forward(self, *inputs):
        # Note: before passed to AttentionModule, x,y has already been preprocessed by conv+BN+ReLU
        x, y = inputs[0], inputs[1]  # [B, c, h, w]
        batch_size = x.size(0)

        u_x = self.gap(x)   # [B, c, 1, 1]
        u_y = self.gap(y)   # [B, c, 1, 1]
        u = torch.cat((u_x, u_y), dim=1)  # [B, 2c, 1, 1]

        z = self.fc1(u)  # [B, d, 1, 1]
        z = self.fc2(z)  # [B, 2c, 1, 1]
        z = z.view(batch_size, 2, self.in_ch, 1, 1)  # [B, 2, c, 1, 1]
        att_score = self.softmax(z)                  # [B, 2, c, 1, 1]

        feats = torch.cat((x,y), dim=1)  # [B, 2c, h, w]
        feats = feats.view(batch_size, 2, self.in_ch, feats.shape[2], feats.shape[3])  # [B, 2, c, h, w]
        feats_V = torch.sum(feats * att_score, dim=1)  # [B, c, h, w]

        out = feats_V if self.M == 2 else feats_V+inputs[2]
        return out


class AttGate3a(nn.Module):
    def __init__(self, in_ch, M=2, r=4):
        # 输入特征的通道数， 2个分支，bottle-net layer的 reduction rate
        super().__init__()
        self.gap = nn.AdaptiveAvgPool2d((1, 1))

        # to calculate Z
        self.fc = nn.Sequential(nn.Conv1d(4, 1, kernel_size=1, bias=False),  # [B, 1, c]
                                nn.BatchNorm1d(1),
                                nn.Sigmoid())

    def forward(self, x, y):
        # Note: before passed to AttentionModule, x,y has already been preprocessed by conv+BN+ReLU

        u_x = self.gap(x).squeeze(-1)         # [B, c, 1]
        u_y = self.gap(y).squeeze(-1)         # [B, c, 1]
        g_x = torch.mean(u_x, 1).unsqueeze(1).expand_as(u_x) # [B, c, 1]
        g_y = torch.mean(u_y, 1).unsqueeze(1).expand_as(u_y) # [B, c, 1]

        m = torch.cat((u_x, u_y, g_x, g_y), dim=2)           # [B, c, 4]
        m = m.permute(0, 2, 1).contiguous()                  # [B, 4, c]
        att = self.fc(m)                                     # [B, 1, c]
        att = att.permute(0, 2, 1).unsqueeze(-1)             # [B, c, 1, 1]
        return x*att+y*(1-att)


class AttGate3b(nn.Module):
    def __init__(self, in_ch, M=2, r=4):
        # 输入特征的通道数， 2个分支，bottle-net layer的 reduction rate
        super().__init__()
        self.gap = nn.AdaptiveAvgPool2d((1, 1))

        # to calculate Z
        self.fc = nn.Sequential(nn.Conv1d(4, 8, kernel_size=1, bias=False),  # [B, 8, c]
                                nn.BatchNorm1d(8),
                                nn.ReLU(inplace=True),
                                nn.Conv1d(8, 1, kernel_size=1, bias=False),  # [B, 1, c]
                                nn.Sigmoid())

    def forward(self, x, y):
        # Note: before passed to AttentionModule, x,y has already been preprocessed by conv+BN+ReLU

        u_x = self.gap(x).squeeze(-1)         # [B, c, 1]
        u_y = self.gap(y).squeeze(-1)         # [B, c, 1]
        g_x = torch.mean(u_x, 1).unsqueeze(1).expand_as(u_x) # [B, c, 1]
        g_y = torch.mean(u_y, 1).unsqueeze(1).expand_as(u_y) # [B, c, 1]

        m = torch.cat((u_x, u_y, g_x, g_y), dim =2)          # [B, c, 4]
        m = m.permute(0, 2, 1).contiguous()                  # [B, 4, c]
        att = self.fc(m)                                     # [B, 1, c]
        att = att.permute(0, 2, 1).unsqueeze(-1)             # [B, c, 1, 1]
        return x*att+y*(1-att)


class AttGate4(nn.Module):
    def __init__(self, hw, in_ch, r=4):
        super().__init__()
        d = max(in_ch//4, 32)
        #
        self.conv1 = nn.Sequential(nn.Conv1d(hw, 32, kernel_size=1, stride=1, bias=True),  # [B, 32, 2c]
                                   nn.BatchNorm1d(32),
                                   nn.ReLU(inplace=True))
        self.conv2 = nn.Conv1d(32, 1, kernel_size=1, stride=1, bias=True)  # [B, 1, 2c]

        self.fc1 = nn.Sequential(nn.Conv2d(in_ch * 2, d, kernel_size=1, stride=1, bias=False),
                                 nn.BatchNorm2d(d),
                                 nn.ReLU(inplace=True))
        self.fc2 = nn.Conv2d(d, in_ch * 2, kernel_size=1, stride=1, bias=False)

    def forward(self, x, y):
        batch_size, ch, h, w = x.shape    # [B, c, h, w]
        m = torch.cat((x,y), dim=1)       # [B, 2c, h, w]
        m = m.view(batch_size, 2*ch, -1).permute(0, 2, 1)  # [B, hw, 2ch]

        gap = self.conv2(self.conv1(m))          # [B, 1, 2c]
        gap = gap.view(batch_size, 2*ch, 1, 1)   # [B, 2c, 1, 1]

        att = self.fc2(self.fc1(gap))            # [B, 2c, 1, 1]
        att = att.view()


class AttGate4c(nn.Module):
    def __init__(self, in_ch, shape=None):
        super().__init__()
        self.h, self.w = shape
        d = max(in_ch//4, 32)

        if self.w > 30:
            self.conv0 = nn.Sequential()
            for i in range(int(np.log2(self.w//30))):
                self.conv0.add_module('conv'+str(i),nn.Conv2d(1, 1, kernel_size=2, stride=2))
        hw = min(30*30, self.h*self.w )

        self.conv1 = nn.Sequential(nn.Conv1d(hw, 8, kernel_size=1, stride=1, bias=True),  # [B, 32, 2c]
                                   nn.BatchNorm1d(8),
                                   nn.ReLU(inplace=True))
        self.conv2 = nn.Conv1d(8, 1, kernel_size=1, stride=1, bias=True)  # [B, 1, 2c]

        self.fc = nn.Sequential(nn.Conv2d(in_ch, d, kernel_size=1, stride=1, bias=False),
                                nn.BatchNorm2d(d),
                                nn.ReLU(inplace=True),
                                nn.Conv2d(d, in_ch, kernel_size=1, stride=1, bias=False),
                                nn.Sigmoid())

    def forward(self, inputs):
        batch_size, ch, h, w = inputs.shape       # [B, c, h, w]
        if self.w > 30:
            x = inputs.view(batch_size*ch, 1, h, w)    # [Bc, 1, h, w]
            x = self.conv0(x)                     # [Bc, 1, 30, 30]
            x = x.view(batch_size, ch, -1)        # [B, c, 30*30]
        else:
            x = inputs.view(batch_size, ch, -1)        # [B, c, hw]
        x = x.permute(0, 2, 1).contiguous()       # [B, hw, c]

        z = self.conv1(x)                         # [B, 8, c]
        z = self.conv2(z)                         # [B, 1, c]
        z = z.view(batch_size, ch, 1, -1)         # [B, c, 1, 1]

        att = self.fc(z)                          # [B, c, 1, 1]
        return att*inputs


class PSPSE(nn.Module):
    def __init__(self, in_ch, r=16, d=None):
        super().__init__()
        int_ch = max(in_ch // r, 8)
        self.pool = nn.AdaptiveAvgPool2d(d)
        self.fc = nn.Sequential(nn.Linear(in_ch*d*d, int_ch, bias=False),
                                nn.ReLU(inplace=True),
                                nn.Linear(int_ch, in_ch, bias=True),
                                nn.Sigmoid())

    def forward(self, x):
        batch_size, ch, h, w = x.size()
        z = self.pool(x)                  # [B, c, d, d]
        z = z.view(batch_size, -1)        # [B, c*d*d]

        z = self.fc(z)                    # [B, c]
        z = z.view(batch_size, ch, 1, 1)  # [B, c, 1, 1]
        return x*z


class AttGate5c(nn.Module):
    def __init__(self, in_ch, r=None):
        super().__init__()
        for d in [1, 2, 4]:
            r = 32 if d == 4 else 16
            self.add_module('att_d{}'.format(d), PSPSE(in_ch=in_ch, r=r, d=d))

    def forward(self, x):
        y1 = self.att_d1(x)
        y2 = self.att_d2(x)
        y4 = self.att_d4(x)

        return y1+y2+y4


class AttGate6(nn.Module):
    def __init__(self, in_ch, shape=None, r=None, update_dep=False, act_fn=None):
        super().__init__()
        # 参考PAN x 为浅层网络，y为深层网络
        self.update_dep = update_dep
        self.x_conv = nn.Sequential(nn.Conv2d(in_ch, in_ch, kernel_size=3, padding=1, bias=False),
                                    nn.BatchNorm2d(in_ch))

        self.y_gap = nn.AdaptiveAvgPool2d(1)
        self.y_conv = nn.Sequential(nn.Conv2d(in_ch, in_ch, kernel_size=1, padding=0, bias=False),
                                    nn.BatchNorm2d(in_ch),
                                    nn.ReLU(inplace=True))
        if act_fn == 'tanh':
            self.activation = nn.Tanh()
        elif act_fn == 'sigmoid':
            self.activation = nn.Sigmoid()
        else:
            self.activation = None


    def forward(self, y, x):
        x1 = self.x_conv(x)      # [B, c, h, w]

        y1 = self.y_gap(y)       # [B, c, 1, 1]
        y1 = self.y_conv(y1)     # [B, c, 1, 1]
        if self.activation:
            y1 = self.activation(y1)

        weighted_x = y1*x1
        if self.update_dep:    # y is rgb, x is dep
            return weighted_x+y, weighted_x
        else:
            return weighted_x+y



class AttGate9(nn.Module):
    # 简单的线性变换
    def __init__(self, in_ch):
        super().__init__()
        self.conv = nn.Conv2d(in_ch*2, in_ch, kernel_size=1, stride=1, groups=in_ch, bias=False)
        self.conv.weight.data.fill_(0.5)

    def forward(self, x, y):
        batch_size, ch, h, w = x.size()
        x1 = x.view(batch_size, ch, -1)                   # [B, c, hw]
        y1 = y.view(batch_size, ch, -1)                   # [B, c, hw]
        m = torch.cat((x1, y1), dim=-1)                   # [B, c, 2hw]
        m = m.view(batch_size, 2*ch, h, -1).contiguous()  # [B, 2c, h, w]
        out = self.conv(m)                                # [B, c, h, w]
        return out
