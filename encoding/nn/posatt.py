#encoding:utf-8
import torch
import torch.nn as nn
from torch.nn import functional as F
from .center import PyramidPooling
from .basic import *

__all__ = ['PosAtt0', 'PosAtt1', 'PosAtt2', 'PosAtt3', 'PosAtt3a', 'PosAtt3c', 'PosAtt4', 'PosAtt4a', 'PosAtt5',
           'PosAtt6', 'PosAtt6a', 'PosAtt7', 'PosAtt7a', 'PosAtt7b', 'PosAtt7d', 'PosAtt9', 'PosAtt9a',
           'CMPA1', 'CMPA1a', 'CMPA2', 'CMPA2a']
up_kwargs = {'mode': 'bilinear', 'align_corners': True}


class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat((torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1)


class PosAtt0(nn.Module):
    def __init__(self, ch, shape=None, r=4, act_fn='sigmoid', conv=None, fuse='add'):
        super(PosAtt0, self).__init__()
        int_ch = max(ch//r, 32)
        self.act_fn, self.conv, self.fuse = act_fn, conv, fuse
        self.W_x = nn.Sequential(nn.Conv2d(ch, int_ch, kernel_size=1, stride=1, padding=0, bias=True),
                                 nn.BatchNorm2d(int_ch))
        self.W_y = nn.Sequential(nn.Conv2d(ch, int_ch, kernel_size=1, stride=1, padding=0, bias=True),
                                 nn.BatchNorm2d(int_ch))
        self.relu = nn.ReLU(inplace=True)

        self.psi = nn.Sequential(nn.Conv2d(int_ch, 1, kernel_size=1, stride=1, padding=0, bias=True),
                                 nn.BatchNorm2d(1),)

        if self.conv == 'conv':
            self.x_conv = nn.Sequential(nn.Conv2d(ch, ch, kernel_size=3, padding=1, bias=False),
                                        nn.BatchNorm2d(ch))
        elif self.conv == 'bblok':
            self.x_conv = BasicBlock(ch, ch)

        if self.fuse == 'cat':
            self.out_conv = nn.Conv2d(ch * 2, ch, kernel_size=1, stride=1)

    def forward(self, y, x):   # 对x(第二个param)进行attention处理
        x1 = self.W_x(x)           # [B, int_c, h, w]
        y1 = self.W_y(y)           # [B, int_c, h, w]
        psi = self.relu(x1 + y1)   # no bias
        psi = self.psi(psi)        # [B, 1, h, w]

        if self.act_fn == 'sigmoid':
            psi = F.sigmoid(psi)
        elif self.act_fn == 'rsigmoid':
            psi = F.sigmoid(F.relu(psi, inplace=True))
        elif self.act_fn == 'tanh':
            psi = F.tanh(F.relu(psi, inplace=True))

        if self.conv:
            x = self.x_conv(x)
        weighted_x = x*psi

        if self.fuse == 'add':
            return weighted_x+y
        elif self.fuse == 'cat':
            return self.out_conv( torch.cat((weighted_x, y), dim=1) )


class PosAtt1(nn.Module):
    def __init__(self, ch, r=4):
        super(PosAtt1, self).__init__()
        int_ch = max(ch//r, 32)
        self.W_x = nn.Sequential(nn.Conv2d(ch, int_ch, kernel_size=1, stride=1, padding=0, bias=True),
                                 nn.BatchNorm2d(int_ch))
        self.W_y = nn.Sequential(nn.Conv2d(ch, int_ch, kernel_size=1, stride=1, padding=0, bias=True),
                                 nn.BatchNorm2d(int_ch))

        self.psi = nn.Sequential(nn.Conv2d(int_ch, 2, kernel_size=1, stride=1, padding=0, bias=True),
                                 nn.BatchNorm2d(2),
                                 nn.Softmax(dim=1))

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, y):
        x1 = self.W_x(x)           # [B, int_c, h, w]
        y1 = self.W_y(y)           # [B, int_c, h, w]
        psi = self.relu(x1 + y1)   # no bias
        psi = self.psi(psi)        # [B, 2, h, w]

        out = x * psi[:, :1].contiguous() + y * psi[:, 1:].contiguous()
        return out


class PosAtt2(nn.Module):
    def __init__(self, in_ch, r=4):
        super(PosAtt2, self).__init__()
        int_ch = max(in_ch//r, 32)
        self.fc = nn.Sequential(nn.Conv2d(2*in_ch, int_ch, kernel_size=1, stride=1, padding=0, bias=True),
                                nn.BatchNorm2d(int_ch),
                                nn.ReLU(inplace=True))

        self.psi = nn.Sequential(nn.Conv2d(int_ch, 2, kernel_size=1, stride=1, padding=0, bias=True),
                                 nn.BatchNorm2d(2),
                                 nn.Softmax(dim=1))

    def forward(self, x, y):
        m = torch.cat((x, y), dim=1)   # [B, 2c, h, w]
        m = self.fc(m)                 # [B, int_c, h, w]
        psi = self.psi(m)              # [B, 2, h, w]

        out = x * psi[:, :1].contiguous() + y * psi[:, 1:].contiguous()
        return out


class PosAtt3(nn.Module):
    def __init__(self, in_ch=None):
        super().__init__()
        kernel_size = 7
        self.compress = ChannelPool()
        self.spatial = nn.Sequential(nn.Conv2d(4, 2, kernel_size=kernel_size, stride=1, padding=(kernel_size-1)//2),
                                     nn.BatchNorm2d(2),
                                     nn.Softmax(dim=1) )

    def forward(self, x, y):
        x1 = self.compress(x)         # [B, 2, h, w]
        y1 = self.compress(y)         # [B, 2, h, w]
        m = torch.cat((x1, y1), dim=1)  # [B, 4, h, w]
        att = self.spatial(m)        # [B, 2, h, w]

        out = x * att[:, :1].contiguous() + y * att[:, 1:].contiguous()
        return out


class PosAtt3c(nn.Module):
    def __init__(self, in_ch=None):
        super().__init__()
        kernel_size = 7
        self.compress = ChannelPool()
        self.spatial = nn.Sequential(nn.Conv2d(2, 1, kernel_size=kernel_size, stride=1, padding=(kernel_size-1)//2),
                                     nn.BatchNorm2d(1),
                                     nn.Sigmoid())

    def forward(self, x):
        x1 = self.compress(x)         # [B, 2, h, w]
        att = self.spatial(x1)        # [B, 1, h, w]

        out = x * att                 # [B, c, h, w]
        return out



class PosAtt3a(nn.Module):
    def __init__(self, in_ch=None):
        super().__init__()
        kernel_size = 7
        self.compress = ChannelPool()
        self.conv_x = nn.Conv2d(in_ch, 2, kernel_size=1)
        self.conv_y = nn.Conv2d(in_ch, 2, kernel_size=1)
        self.spatial = nn.Sequential(nn.Conv2d(8, 2, kernel_size=kernel_size, stride=1, padding=(kernel_size-1)//2),
                                     nn.BatchNorm2d(2),
                                     nn.Softmax(dim=1) )

    def forward(self, x, y):
        x1 = self.compress(x)         # [B, 2, h, w]
        y1 = self.compress(y)         # [B, 2, h, w]
        x2 = self.conv_x(x)           # [B, 2, h, w]
        y2 = self.conv_y(y)           # [B, 2, h, w]

        m = torch.cat((x1, y1, x2, y2), dim=1)  # [B, 8, h, w]
        att = self.spatial(m)        # [B, 2, h, w]
        out = x * att[:, :1].contiguous() + y * att[:, 1:].contiguous()
        return out


class PosAtt4(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        int_ch = 4
        self.f_conv = nn.Sequential(nn.Conv2d(2*in_ch, int_ch, kernel_size=1, stride=1),
                                    nn.BatchNorm2d(int_ch))
        self.compress = ChannelPool()
        self.pool1 = nn.AdaptiveAvgPool2d(1)
        self.pool2 = nn.AdaptiveAvgPool2d(2)
        self.pool3 = nn.AdaptiveAvgPool2d(3)
        self.pool6 = nn.AdaptiveAvgPool2d(6)

        self.psi = nn.Sequential(nn.Conv2d(5*(int_ch+4), 2, kernel_size=1, stride=1, padding=0, bias=True),
                                 nn.BatchNorm2d(2),
                                 nn.Softmax(dim=1))

    def forward(self, x, y):
        _, _, h, w = x.size()
        xy = torch.cat((x, y), dim=1)       # [B, 2c, h, w]
        m1 = self.f_conv(xy)                # [B, 4, h, w]
        x1, y1 = self.compress(x), self.compress(y)
        m = torch.cat((m1, x1, y1), dim=1)  # [B, 4+4, h, w]

        p1 = F.interpolate(self.pool1(m), (h, w), **up_kwargs)
        p2 = F.interpolate(self.pool2(m), (h, w), **up_kwargs)
        p3 = F.interpolate(self.pool3(m), (h, w), **up_kwargs)
        p6 = F.interpolate(self.pool6(m), (h, w), **up_kwargs)

        z = torch.cat((m, p1, p2, p3, p6), dim=1)  # [B, (4+4)*5, h, w]
        att = self.psi(z)                          # [B, 2, h, w]
        out = x * att[:, :1].contiguous() + y * att[:, 1:].contiguous()
        return out


class PosAtt4a(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        int_ch = 4
        self.f_conv = nn.Sequential(nn.Conv2d(2 * in_ch, int_ch, kernel_size=1, stride=1),
                                    nn.BatchNorm2d(int_ch))
        self.compress = ChannelPool()
        self.ppm = PyramidPooling(in_channels=int_ch+4, norm_layer=nn.BatchNorm2d, up_kwargs=up_kwargs)

        self.psi = nn.Sequential(nn.Conv2d(2 * (int_ch + 4), 2, kernel_size=1, stride=1, padding=0, bias=True),
                                 nn.BatchNorm2d(2),
                                 nn.Softmax(dim=1))

    def forward(self, x, y):
        _, _, h, w = x.size()
        xy = torch.cat((x, y), dim=1)  # [B, 2c, h, w]
        m1 = self.f_conv(xy)  # [B, 4, h, w]
        x1, y1 = self.compress(x), self.compress(y)
        m = torch.cat((m1, x1, y1), dim=1)  # [B, 4+4, h, w]

        z = self.ppm(m)  # [B, (4+4)*2, h, w]
        att = self.psi(z)  # [B, 2, h, w]
        out = x * att[:, :1].contiguous() + y * att[:, 1:].contiguous()
        return out


class PosAtt5(nn.Module):
    def __init__(self, in_ch, r=10, w=240):
        super().__init__()
        self.d = min(15, w//3)
        int_n = max(self.d*self.d//r, 20)
        self.pool = nn.AdaptiveAvgPool2d((self.d, self.d))

        self.f_conv = nn.Sequential(nn.Conv2d(2 * in_ch, 1, kernel_size=1, stride=1),
                                    nn.BatchNorm2d(1))
        self.fc1 = nn.Sequential(nn.Linear(self.d*self.d, int_n),
                                 nn.BatchNorm1d(int_n),
                                 nn.ReLU(inplace=True))
        self.fc2 = nn.Linear(int_n, self.d*self.d*2)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x, y):
        batch_size, ch, w, h = x.size()
        m = torch.cat((x,y), dim=1)  # [B, 2c, h, w]
        m = self.pool(m)             # [B, 2c, d, d]
        m = self.f_conv(m).view(batch_size, -1)    # [B, d*d]

        z = self.fc1(m)                             # [B, int_n]
        z = self.fc2(z)                             # [B, d*d*2]
        z = z.view(batch_size, -1, self.d, self.d)  # [B, 2, d, d]

        z = F.interpolate(z, (h, w), **up_kwargs)   # [B, 2, h, w]
        att = self.softmax(z)                       # [B, 2, h, w]
        out = x * att[:, :1].contiguous() + y * att[:, 1:].contiguous()
        return out


class PosAtt6(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        self.x_conv = nn.Conv2d(in_ch, 1, kernel_size=1)
        self.y_conv = nn.Conv2d(in_ch, 1, kernel_size=1)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x, y):
        x1 = self.x_conv(x)             # [B, 1, h, w]
        y1 = self.y_conv(y)             # [B, 1, h, w]
        m = torch.cat((x1, y1), dim=1)  # [B, 2, h, w]
        att = self.softmax(m)           # [B, 2, h, w]

        out = x*att[:, :1].contiguous() + y*att[:, 1:].contiguous()
        return out


class PosAtt6a(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        F = 7
        self.x_conv = nn.Conv2d(in_ch, 1, kernel_size=1)
        self.y_conv = nn.Conv2d(in_ch, 1, kernel_size=1)
        self.m_conv = nn.Conv2d(2, 2, kernel_size=7, stride=1, padding=(F - 1) // 2)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x, y):
        x1 = self.x_conv(x)             # [B, 1, h, w]
        y1 = self.y_conv(y)             # [B, 1, h, w]
        m = torch.cat((x1, y1), dim=1)  # [B, 2, h, w]
        m = self.m_conv(m)              # [B, 2, h, w] 参考周围点
        att = self.softmax(m)           # [B, 2, h, w]

        out = x * att[:, :1].contiguous() + y * att[:, 1:].contiguous()
        return out


class PosAtt7(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        self.y_conv = nn.Sequential(nn.Conv2d(in_ch, 1, kernel_size=1),
                                    nn.Sigmoid())

    def forward(self, x, y):
        # x is dep, y is rgb.  x 浅层网络特征， y为深层网络特征
        y1 = self.y_conv(y)     # [B, 1, h, w]
        x1 = torch.mul(x, y1)
        return x1 + y


class PosAtt7a(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        self.y_conv = nn.Sequential(nn.Conv2d(in_ch, 1, kernel_size=1, bias=False),
                                    nn.BatchNorm2d(1),
                                    nn.Sigmoid())
        self.x_conv = nn.Sequential(nn.Conv2d(in_ch, in_ch, 3, padding=1, bias=False),
                                    nn.BatchNorm2d(in_ch))

    def forward(self, x, y):
        # x is dep, y is rgb.  x 浅层网络特征， y为深层网络特征
        y1 = self.y_conv(y)     # [B, 1, h, w] weight
        x1 = self.x_conv(x)     # [B, c, h, w]
        out = torch.mul(x1, y1) + y
        return out


class PosAtt7b(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        self.y_conv = nn.Conv2d(in_ch, 1, kernel_size=1, bias=False)
        self.x_conv = nn.Sequential(nn.Conv2d(in_ch, in_ch, 3, padding=1, bias=False),
                                    nn.BatchNorm2d(in_ch))

    def forward(self, x, y):
        # x is dep, y is rgb.  x 浅层网络特征， y为深层网络特征
        y1 = self.y_conv(y)  # [B, 1, h, w] weight
        y_bn = nn.LayerNorm(y1.size()[1:])
        y1 = F.sigmoid(y_bn(y1))  # [B, 1, h, w] 归一化并进行sigmoid

        x1 = self.x_conv(x)  # [B, c, h, w]
        out = torch.mul(x1, y1) + y
        return out


class PosAtt7d(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        self.y_conv = nn.Sequential(nn.Conv2d(in_ch, 1, kernel_size=1, bias=False),
                                    nn.BatchNorm2d(1),
                                    nn.ReLU(inplace=True))
        self.x_conv = nn.Sequential(nn.Conv2d(in_ch, in_ch, 3, padding=1, bias=False),
                                    nn.BatchNorm2d(in_ch))

    def forward(self, x, y):
        # x is dep, y is rgb.  x 浅层网络特征， y为深层网络特征
        y1 = self.y_conv(y)     # [B, 1, h, w] weight
        x1 = self.x_conv(x)     # [B, c, h, w]
        out = torch.mul(x1, y1) + y
        return out



class PosAtt9(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, 1, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        batch_size, ch, h, w = x.size()
        z = self.conv(x)       # [B, c, h, w]
        z = self.sigmoid(z)    # [B, c, h, w]
        out = torch.mul(x, z.view(batch_size, 1, h, w))
        return out


class PosAtt9a(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        self.conv = nn.Sequential(nn.Conv2d(in_ch, 1, kernel_size=1),
                                  nn.Sigmoid())
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.conv_gap = nn.Sequential(nn.Conv2d(in_ch, in_ch, kernel_size=1, bias=False),
                                      nn.BatchNorm2d(in_ch))

    def forward(self, x):
        batch_size, ch, h, w = x.size()
        z = self.conv(x)       # [B, c, h, w]
        out = torch.mul(x, z.view(batch_size, 1, h, w))

        x_gap = self.gap(x).view(batch_size, ch, 1, 1)   # [B, c, 1, 1]
        x_gap = self.conv_gap(x_gap)                     # [B, c, 1, 1]
        return out+x_gap



class CMPA1(nn.Module):
    def __init__(self, in_ch=None, shape=None):
        super().__init__()
        h, w =shape
        self.ap1 = nn.AdaptiveAvgPool2d(1)
        self.ap2 = nn.AdaptiveAvgPool2d(2)
        self.ap3 = nn.AdaptiveAvgPool2d(3)
        self.ap6 = nn.AdaptiveAvgPool2d(6)

        self.fc = nn.Sequential(nn.Linear(1+4+9+36, w*w),
                                nn.BatchNorm1d(w*w),
                                nn.ReLU(inplace=True))

    def forward(self, y, x):
        # x is dep, y is rgb.  x 浅层网络特征， y为深层网络特征
        batch_size, ch, h, w = y.size()
        y1 = torch.mean(y, 1).unsqueeze(1)         # [B, 1, h, w]
        y1_1 = self.ap1(y1).view(batch_size, 1)    # [B, 1, 1, 1] -> [B, 1]
        y1_2 = self.ap2(y1).view(batch_size, 4)    # [B, 1, 2, 2] -> [B, 4]
        y1_3 = self.ap3(y1).view(batch_size, 9)
        y1_6 = self.ap6(y1).view(batch_size, 36)

        y2 = torch.cat((y1_1, y1_2, y1_3, y1_6), dim=1)
        att = self.fc(y2).view(batch_size, -1, h, w).contiguous()   # [B, w*w] -> [B, 1, h, w]

        x1 = torch.mul(x, att)
        return x1 + y


class CMPA1a(nn.Module):
    def __init__(self, in_ch=None, shape=None):
        super().__init__()
        h, w = shape
        self.ap1 = nn.AdaptiveAvgPool2d(1)
        self.ap2 = nn.AdaptiveAvgPool2d(2)
        self.ap3 = nn.AdaptiveAvgPool2d(3)
        self.ap6 = nn.AdaptiveAvgPool2d(6)

        self.fc = nn.Sequential(nn.Linear(1 + 4 + 9 + 36, w * w),
                                nn.BatchNorm1d(w * w),
                                nn.Sigmoid())

    def forward(self, y, x):
        # x is dep, y is rgb.  x 浅层网络特征， y为深层网络特征
        batch_size, ch, h, w = y.size()
        y1 = torch.mean(y, 1).unsqueeze(1)  # [B, 1, h, w]
        y1_1 = self.ap1(y1).view(batch_size, 1)  # [B, 1, 1, 1] -> [B, 1]
        y1_2 = self.ap2(y1).view(batch_size, 4)  # [B, 1, 2, 2] -> [B, 4]
        y1_3 = self.ap3(y1).view(batch_size, 9)
        y1_6 = self.ap6(y1).view(batch_size, 36)

        y2 = torch.cat((y1_1, y1_2, y1_3, y1_6), dim=1)
        att = self.fc(y2).view(batch_size, -1, h, w).contiguous()  # [B, w*w] -> [B, 1, h, w]

        x1 = torch.mul(x, att)
        return x1 + y



class CMPA2(nn.Module):
    def __init__(self, in_ch=None, shape=None):
        super().__init__()
        h, w = shape
        self.ap1 = nn.AdaptiveAvgPool2d(1)
        self.ap2 = nn.AdaptiveAvgPool2d(2)
        self.ap3 = nn.AdaptiveAvgPool2d(3)
        self.ap5 = nn.AdaptiveAvgPool2d(5)
        self.ap10 = nn.AdaptiveAvgPool2d(10) if w>15 else None
        self.ap15 = nn.AdaptiveAvgPool2d(15) if w>30 else None
        self.ap30 = nn.AdaptiveAvgPool2d(30) if w>60 else None

        ch = sum([1 if i is not None else 0 for i in [self.ap10, self.ap15, self.ap30]]) + 5
        self.fc = nn.Sequential(nn.Conv2d(ch, 1, kernel_size=1),    # [B, 1, h, w]
                                nn.BatchNorm2d(1),
                                nn.ReLU(inplace=True))

    def forward(self, y, x):
        # x is dep, y is rgb.  x 浅层网络特征， y为深层网络特征
        batch_size, ch, h, w = y.size()
        y1 = torch.mean(y, 1).unsqueeze(1)  # [B, 1, h, w]
        y2 = y1
        for ap in [self.ap1, self.ap2, self.ap3, self.ap5, self.ap10, self.ap15, self.ap30]:
            if ap:
                p = ap(y1)
                p = F.interpolate(p, (h, w), **up_kwargs)   # [B, 1, h, w]
                y2 = torch.cat((y2, p), dim=1)
        y2 = y2.contiguous()
        att = self.fc(y2)      # [B, 1, h, w]

        x1 = torch.mul(x, att)
        return x1 + y


class CMPA2a(nn.Module):
    def __init__(self, in_ch=None, shape=None):
        super().__init__()
        h, w = shape
        self.ap1 = nn.AdaptiveAvgPool2d(1)
        self.ap2 = nn.AdaptiveAvgPool2d(2)
        self.ap3 = nn.AdaptiveAvgPool2d(3)
        self.ap5 = nn.AdaptiveAvgPool2d(5)
        self.ap10 = nn.AdaptiveAvgPool2d(10) if w>15 else None
        self.ap15 = nn.AdaptiveAvgPool2d(15) if w>30 else None
        self.ap30 = nn.AdaptiveAvgPool2d(30) if w>60 else None

        ch = sum([1 if i is not None else 0 for i in [self.ap10, self.ap15, self.ap30]]) + 5
        self.fc = nn.Sequential(nn.Conv2d(ch, 1, kernel_size=1),    # [B, 1, h, w]
                                nn.BatchNorm2d(1),
                                nn.Sigmoid())

    def forward(self, y, x):
        # x is dep, y is rgb.  x 浅层网络特征， y为深层网络特征
        batch_size, ch, h, w = y.size()
        y1 = torch.mean(y, 1).unsqueeze(1)  # [B, 1, h, w]
        y2 = y1
        for ap in [self.ap1, self.ap2, self.ap3, self.ap5, self.ap10, self.ap15, self.ap30]:
            if ap:
                p = ap(y1)
                p = F.interpolate(p, (h, w), **up_kwargs)   # [B, 1, h, w]
                y2 = torch.cat((y2, p), dim=1)
        y2 = y2.contiguous()
        att = self.fc(y2)      # [B, 1, h, w]

        x1 = torch.mul(x, att)
        return x1 + y

