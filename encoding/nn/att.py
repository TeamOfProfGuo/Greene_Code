import torch
from torch import nn

__all__ = ['ContextBlock', 'FPA']

def constant_init(module, val, bias=0):
    if hasattr(module, 'weight') and module.weight is not None:
        nn.init.constant_(module.weight, val)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)

def kaiming_init(module,a=0,mode='fan_out',nonlinearity='relu',bias=0, distribution='normal'):
    assert distribution in ['uniform', 'normal']
    if hasattr(module, 'weight') and module.weight is not None:
        if distribution == 'uniform':
            nn.init.kaiming_uniform_(module.weight, a=a, mode=mode, nonlinearity=nonlinearity)
        else:
            nn.init.kaiming_normal_(module.weight, a=a, mode=mode, nonlinearity=nonlinearity)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)


def last_zero_init(m):
    if isinstance(m, nn.Sequential):
        constant_init(m[-1], val=0)
    else:
        constant_init(m, val=0)


class ContextBlock(nn.Module):

    def __init__(self,in_ch, r=8, fusion_types=('channel_add', )):
        super(ContextBlock, self).__init__()
        assert isinstance(fusion_types, (list, tuple))
        valid_fusion_types = ['channel_add', 'channel_mul']
        assert all([f in valid_fusion_types for f in fusion_types])
        assert len(fusion_types) > 0, 'at least one fusion should be used'
        self.inplanes = in_ch
        self.ratio = r
        self.planes = int(in_ch * r)
        self.fusion_types = fusion_types

        self.conv_mask = nn.Conv2d(in_ch, 1, kernel_size=1)
        self.softmax = nn.Softmax(dim=2)

        if 'channel_add' in fusion_types:
            self.channel_add_conv = nn.Sequential(
                nn.Conv2d(self.inplanes, self.planes, kernel_size=1),
                nn.LayerNorm([self.planes, 1, 1]),
                nn.ReLU(inplace=True),  # yapf: disable
                nn.Conv2d(self.planes, self.inplanes, kernel_size=1))
        else:
            self.channel_add_conv = None
        if 'channel_mul' in fusion_types:
            self.channel_mul_conv = nn.Sequential(
                nn.Conv2d(self.inplanes, self.planes, kernel_size=1),
                nn.LayerNorm([self.planes, 1, 1]),
                nn.ReLU(inplace=True),  # yapf: disable
                nn.Conv2d(self.planes, self.inplanes, kernel_size=1))
        else:
            self.channel_mul_conv = None
        self.reset_parameters()

    def reset_parameters(self):
        if self.channel_add_conv is not None:
            last_zero_init(self.channel_add_conv)
        if self.channel_mul_conv is not None:
            last_zero_init(self.channel_mul_conv)

    def spatial_pool(self, x):
        batch, channel, height, width = x.size()
        input_x = x
        input_x = input_x.view(batch, channel, height * width) # [N, C, H * W]
        input_x = input_x.unsqueeze(1)                         # [N, 1, C, H * W]
        context_mask = self.conv_mask(x)                       # [N, 1, H, W]
        context_mask = context_mask.view(batch, 1, height * width)  # [N, 1, H * W]
        context_mask = self.softmax(context_mask)              # [N, 1, H * W]
        context_mask = context_mask.unsqueeze(-1)              # [N, 1, H * W, 1]

        context = torch.matmul(input_x, context_mask)          # [N, 1, C, 1]
        context = context.view(batch, channel, 1, 1)           # [N, C, 1, 1]
        return context

    def forward(self, x):
        context = self.spatial_pool(x)    # [N, C, 1, 1]

        out = x
        if self.channel_mul_conv is not None:
            channel_mul_term = torch.sigmoid(self.channel_mul_conv(context)) # [N, C, 1, 1]
            out = out * channel_mul_term
        if self.channel_add_conv is not None:
            channel_add_term = self.channel_add_conv(context)                # [N, C, 1, 1]
            out = out + channel_add_term

        return out


class FPA(nn.Module):
    def __init__(self, in_ch=2048):
        """Feature Pyramid Attention"""
        super().__init__()
        int_ch = int(in_ch/4)
        self.ch = in_ch

        # Master branch
        self.conv_master = nn.Sequential(nn.Conv2d(in_ch, in_ch, kernel_size=1, bias=False),
                                         nn.BatchNorm2d(in_ch))

        # Global pooling branch
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.conv_gpb = nn.Sequential(nn.Conv2d(self.ch, in_ch, kernel_size=1, bias=False),
                                      nn.BatchNorm2d(in_ch))

        # C333 because of the shape of last feature maps is (16, 16).
        self.conv7x7_1 = nn.Sequential(nn.Conv2d(in_ch, int_ch, kernel_size=(7, 7), stride=2, padding=3, bias=False),
                                       nn.BatchNorm2d(int_ch))
        self.conv5x5_1 = nn.Sequential(nn.Conv2d(int_ch, int_ch, kernel_size=(5, 5), stride=2, padding=2, bias=False),
                                       nn.BatchNorm2d(int_ch))
        self.conv3x3_1 = nn.Sequential(nn.Conv2d(int_ch, int_ch, kernel_size=(3, 3), stride=2, padding=1, bias=False),
                                       nn.BatchNorm2d(int_ch))

        self.conv7x7_2 = nn.Sequential(nn.Conv2d(int_ch, int_ch, kernel_size=(7, 7), stride=1, padding=3, bias=False),
                                       nn.BatchNorm2d(int_ch))
        self.conv5x5_2 = nn.Sequential(nn.Conv2d(int_ch, int_ch, kernel_size=(5, 5), stride=1, padding=2, bias=False),
                                       nn.BatchNorm2d(int_ch))
        self.conv3x3_2 = nn.Sequential(nn.Conv2d(int_ch, int_ch, kernel_size=(3, 3), stride=1, padding=1, bias=False),
                                       nn.BatchNorm2d(int_ch))

        # Convolution Upsample
        self.conv_upsample_3 = nn.Sequential(nn.ConvTranspose2d(int_ch, int_ch, kernel_size=4, stride=2, padding=1, bias=False),
                                             nn.BatchNorm2d(int_ch))
        self.conv_upsample_2 = nn.Sequential(nn.ConvTranspose2d(int_ch, int_ch, kernel_size=4, stride=2, padding=1, bias=False),
                                             nn.BatchNorm2d(int_ch))
        self.conv_upsample_1 = nn.Sequential(nn.ConvTranspose2d(int_ch, in_ch, kernel_size=4, stride=2, padding=1, bias=False),
                                             nn.BatchNorm2d(in_ch))

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        """
        :param x: Shape: [b, 2048, h, w]
        :return: out: Feature maps. Shape: [b, 2048, h, w]
        """
        # Master branch
        x_master = self.conv_master(x)    # [B, c, h, w]

        # Global pooling branch
        x_gpb = self.gap(x).view(x.shape[0], self.ch, 1, 1)
        x_gpb = self.conv_gpb(x_gpb)

        # Branch 1
        x1_1 = self.conv7x7_1(x)
        x1_1 = self.relu(x1_1)
        x1_2 = self.conv7x7_2(x1_1)

        # Branch 2
        x2_1 = self.conv5x5_1(x1_1)
        x2_1 = self.relu(x2_1)
        x2_2 = self.conv5x5_2(x2_1)

        # Branch 3
        x3_1 = self.conv3x3_1(x2_1)
        x3_1 = self.relu(x3_1)
        x3_2 = self.conv3x3_2(x3_1)

        # Merge branch 1 and 2
        x3_upsample = self.relu(self.conv_upsample_3(x3_2))
        x2_merge = self.relu(x2_2 + x3_upsample)
        x2_upsample = self.relu(self.conv_upsample_2(x2_merge))
        x1_merge = self.relu(x1_2 + x2_upsample)

        x_master = x_master * self.relu(self.conv_upsample_1(x1_merge))
        out = self.relu(x_master + x_gpb)

        return out
