###########################################################################
# Created by: Hang Zhang
# Email: zhang.hang@rutgers.edu
# Copyright (c) 2017
###########################################################################

import os, sys
BASE_DIR = '/scratch/lg154/sseg/Greene_Code/'
sys.path.append(BASE_DIR)
import copy
import yaml
import logging
import argparse
import numpy as np
from tqdm import tqdm
from addict import Dict

import torch
import torch.nn as nn
from torch.utils import data
from tensorboardX import SummaryWriter
import torchvision.transforms as transform
from torch.nn.parallel.scatter_gather import gather

import encoding.utils as utils
from encoding.nn import SegmentationLosses
from encoding.datasets import get_dataset
from encoding.models import get_segmentation_model

BASE_DIR = '.'
CONFIG_PATH = '/scratch/lg154/sseg/Greene_Code/sun_experiments/irb_psk_pdl_wt1/results/config.yaml'
SMY_PATH = os.path.dirname(CONFIG_PATH)
GPUS = [0,1]

s = 'hf_0002'
model_kwargs = utils.get_model_args(s)
model_kwargs = {k:v for k, v in model_kwargs.items() if v is not None}
print(model_kwargs)

train_setting = utils.split_train_args(s)
train_args = utils.get_train_args(train_setting)
print(train_args)

# configuration
args = Dict(yaml.safe_load(open(CONFIG_PATH)))
args.cuda = (args.use_cuda and torch.cuda.is_available())
torch.manual_seed(args.seed)
args.batch_size = 2

# ================= trainer init  ======================
# data transforms
input_transform = transform.Compose([
    transform.ToTensor(),  # convert RGB [0,255] to FloatTensor in range [0, 1]
    transform.Normalize([.485, .456, .406], [.229, .224, .225])])   # mean and std based on imageNet
dep_transform = transform.Compose([
    transform.ToTensor(),
    transform.Normalize(mean=[0.2798], std=[0.1387])  # mean and std for depth
])
# dataset
data_kwargs = {'transform': input_transform, 'dep_transform':dep_transform,
               'base_size': args.base_size, 'crop_size': args.crop_size}
trainset = get_dataset(args.dataset, split=args.train_split, mode='train', **data_kwargs)
testset = get_dataset(args.dataset, split='val', mode='val', **data_kwargs)

# dataloader
kwargs = {'num_workers': args.workers, 'pin_memory': True} if args.cuda else {}
trainloader = data.DataLoader(trainset, batch_size=args.batch_size, drop_last=True, shuffle=True, **kwargs)
valloader = data.DataLoader(testset, batch_size=args.batch_size, drop_last=False, shuffle=False, **kwargs)
nclass = trainset.num_class

# model
root = '/scratch/lg154/sseg/Greene_Code/encoding/models/pretrain'
model = get_segmentation_model(args.model, dataset=args.dataset, backbone=args.backbone, pretrained=True, dtype = args.dtype,
                               root=root, **model_kwargs)

print(model)


# using cuda
device = torch.device("cuda:0" if args.cuda else "cpu")
if args.cuda:
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(),
              "GPUs!")  # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
        model = nn.DataParallel(model, device_ids=GPUS)
model = model.to(device)


# optimizer using different LR
base_modules = [model.base, model.d_layer1, model.d_layer2, model.d_layer3, model.d_layer4]
base_ids = utils.get_param_ids(base_modules)
base_params = filter(lambda p: id(p) in base_ids, model.parameters())
other_params = filter(lambda p: id(p) not in base_ids, model.parameters())
optimizer = torch.optim.SGD([{'params': base_params, 'lr': args.lr},
                             {'params': other_params, 'lr': args.lr*10}],
                            lr=args.lr,momentum=args.momentum, weight_decay=args.weight_decay)

# criterions

import pickle
fname = 'wt'+str(train_args['class_weight'][0])+'.pickle'
with open(os.path.join('/scratch/lg154/sseg/dataset/NYUD_v2/weight', fname), 'rb') as handle:
    wt = pickle.load(handle)
class_wt = torch.FloatTensor(wt)
# class_wt = torch.FloatTensor(wt).to(device)

type = None if len(train_args['class_weight'])==1 else 's'
criterion = SegmentationLosses(aux=model_kwargs.get('aux'),
                                    nclass=nclass, weight=class_wt,
                                    aux_weight=train_args['aux_weight'], type=type)


scheduler = utils.LR_Scheduler_Head(args.lr_scheduler, args.lr, args.epochs, len(trainloader), warmup_epochs=1)
best_pred = 0.0





# ==================== train =====================
train_loss = 0.0
epoch = 1
print(image)
print(image)
model.train()
for i, (image, dep, target) in enumerate(trainloader):
    print('1 batch')
    break

scheduler(optimizer, i, epoch, best_pred)

optimizer.zero_grad()


outputs = model(image, dep)

loss = criterion(*outputs, target)
loss.backward()
optimizer.step()

train_loss += loss.item()

print('------+++++++++')
print(train_loss)







data_dir = '/scratch/lg154/sseg/dataset/sunrgbd' 

img_dir = ['SUNRGBD/kv2/kinect2data/000065_2014-05-16_20-14-38_260595134347_rgbf000121-resize/image/0000121.jpg',
           'SUNRGBD/kv2/kinect2data/000066_2014-04-13_23-39-40_094959634447_rgbf000225-resize/image/0000225.jpg',
           'SUNRGBD/kv2/kinect2data/000067_2014-04-13_23-40-52_094959634447_rgbf000225-resize/image/0000225.jpg', ]
dep_dir = ['SUNRGBD/kv2/kinect2data/000065_2014-05-16_20-14-38_260595134347_rgbf000121-resize/depth_bfx/0000121.png',
           'SUNRGBD/kv2/kinect2data/000066_2014-04-13_23-39-40_094959634447_rgbf000225-resize/depth_bfx/0000225.png',
           'SUNRGBD/kv2/kinect2data/000067_2014-04-13_23-40-52_094959634447_rgbf000225-resize/depth_bfx/0000225.png', ]
t_dir =  ['SUNRGBD/kv2/kinect2data/000065_2014-05-16_20-14-38_260595134347_rgbf000121-resize/label/label.npy',
          'SUNRGBD/kv2/kinect2data/000066_2014-04-13_23-39-40_094959634447_rgbf000225-resize/label/label.npy',
          'SUNRGBD/kv2/kinect2data/000067_2014-04-13_23-40-52_094959634447_rgbf000225-resize/label/label.npy', ]

i = 0
fpath = os.path.join(data_dir, img_dir[0])
from matplotlib import pyplot as plt
from PIL import Image
image = Image.open(fpath)
image.show()


self = trainset
idx = 0 
_img = self.load_image(idx)
_dep = self.load_depth(idx)
_target = self.load_target(idx)

# synchronized transform
if self.mode == 'train':
    # return _img (Image), _dep (Image), _target (2D tensor)
    _img, _dep, _target = self._sync_transform(_img, _target, depth=_dep,
                                                IGNORE_LABEL=0)  # depth need to modify

_target -= 1

# general resize, normalize and toTensor
if self.transform is not None:
    _img = self.transform(_img)  # _img to tensor, normalize
if self.dep_transform is not None:
    _dep = self.dep_transform(_dep)  # depth to tensor, normalize
if self.target_transform is not None:
    _target = self.target_transform(_target)
return _img, _dep, _target  # all tensors