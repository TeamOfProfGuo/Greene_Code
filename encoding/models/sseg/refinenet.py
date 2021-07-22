import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from ...nn import ResidualConvUnit, MultiResolutionFusion, ChainedResidualPool

__all__ = ['RefineNet', 'get_refinenet']


class RefineNet(nn.Module):
    def __init__(self, n_classes=21, backbone='resnet18', pretrained=True, root='./encoding/models/pretrain',
                 n_features=256, with_CRP=True, with_dep=False):
        super(RefineNet, self).__init__()
        self.do = nn.Dropout(p=0.5)
        self.base = models.resnet18(pretrained=False)
        if with_dep:
            self.base.conv1 = nn.Conv2d(4, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)

        if pretrained:
            if backbone == 'resnet18':
                f_path = os.path.abspath(os.path.join(root, 'resnet18-5c106cde.pth'))
            if not os.path.exists(f_path):
                raise FileNotFoundError('the pretrained model can not be found')
            weights = torch.load(f_path)

            if with_dep:
                new_wt = torch.normal(mean=0, std=0.1, size=(64, 4, 7, 7))
                new_wt[:, 0:3, :, :] = weights['conv1.weight']
                weights['conv1.weight'] = new_wt

            self.base.load_state_dict(weights, strict=False)

        self.in_block = nn.Sequential(self.base.conv1, self.base.bn1, self.base.relu, self.base.maxpool)  # [B, 64, h/4, w/4]

        self.layer1 = self.base.layer1  # [B, 64, h/4, w/4]
        self.layer2 = self.base.layer2  # [B, 128, h/8, w/8]
        self.layer3 = self.base.layer3  # [B, 256, h/16, w/16]
        self.layer4 = self.base.layer4  # [B, 512, h/32, w/32]

        self.layer4_rn = nn.Conv2d(512, 2*n_features, kernel_size=3, stride=1, padding=1, bias=False)
        self.layer3_rn = nn.Conv2d(256, n_features, kernel_size=3, stride=1, padding=1, bias=False)
        self.layer2_rn = nn.Conv2d(128, n_features, kernel_size=3, stride=1, padding=1, bias=False)
        self.layer1_rn = nn.Conv2d(64,  n_features, kernel_size=3, stride=1, padding=1, bias=False)

        self.refine4 = RefineNetBlock(2*n_features, [(2*n_features, 32)], with_CRP=with_CRP)
        self.refine3 = RefineNetBlock(n_features, [(2*n_features, 32), (n_features, 16)], with_CRP=with_CRP)
        self.refine2 = RefineNetBlock(n_features, [(n_features, 16), (n_features, 8)], with_CRP=with_CRP)
        self.refine1 = RefineNetBlock(n_features, [(n_features, 8), (n_features, 4)], with_CRP=with_CRP)

        self.out_conv = nn.Sequential(
            ResidualConvUnit(n_features), ResidualConvUnit(n_features),
            nn.Conv2d(n_features, n_classes, kernel_size=1, stride=1, padding=0, bias=True))

    def forward(self, x):
        _, _, h, w = x.size()
        x = self.in_block(x)

        l1 = self.layer1(x)
        l2 = self.layer2(l1)
        l3 = self.layer3(l2)  # [B, 256, h/16, w/16]
        l4 = self.layer4(l3)  # [B, 512, h/32, w/32]

        # l4 = self.do(l4)
        # l3 = self.do(l3)
        l1 = self.layer1_rn(l1)
        l2 = self.layer2_rn(l2)
        l3 = self.layer3_rn(l3)
        l4 = self.layer4_rn(l4)  # [B, 512, h/32, w/32]

        path4 = self.refine4(l4)          # [B, 512, h/32, w/32]
        path3 = self.refine3(path4, l3)   # [B, 256, h/16, w/16]
        path2 = self.refine2(path3, l2)   # [B, 256, h/8, w/8]
        path1 = self.refine1(path2, l1)

        out = self.out_conv(path1)
        out = F.interpolate(out, (h, w), mode='bilinear', align_corners=True)
        return out


def get_refinenet(dataset='nyud', backbone='resnet34', pretrained=True, root='./encoding/models/pretrain',
                  n_features=256, with_CRP=True, with_dep=False):
    from ...datasets import datasets
    model = RefineNet(datasets[dataset.lower()].NUM_CLASS, backbone=backbone, pretrained=pretrained, root=root,
                      n_features=n_features, with_CRP=with_CRP, with_dep=with_dep)
    return model


class BaseRefineNetBlock(nn.Module):
    def __init__(self, features, residual_conv_unit, multi_resolution_fusion, chained_residual_pool, shapes, with_CRP=True):
        super().__init__()
        self.with_CRP = with_CRP
        for i, shape in enumerate(shapes):  # [(n_features1, scale1), (n_features2, scale2)]
            feats = shape[0]
            self.add_module("rcu{}".format(i), nn.Sequential(residual_conv_unit(feats), residual_conv_unit(feats)))

        self.mrf = multi_resolution_fusion(features, shapes) if len(shapes) != 1 else None

        self.crp = chained_residual_pool(features)
        self.output_conv = residual_conv_unit(features)

    def forward(self, *xs):
        rcu_xs = []

        for i, x in enumerate(xs):
            rcu_xs.append(self.__getattr__("rcu{}".format(i))(x))

        if self.mrf is not None:
            out = self.mrf(*rcu_xs)
        else:
            out = rcu_xs[0]

        if self.with_CRP:
            out = self.crp(out)
        return self.output_conv(out)


class RefineNetBlock(BaseRefineNetBlock):
    def __init__(self, features, shapes, with_CRP=False):
        super().__init__(features, ResidualConvUnit, MultiResolutionFusion, ChainedResidualPool, shapes, with_CRP=with_CRP)
