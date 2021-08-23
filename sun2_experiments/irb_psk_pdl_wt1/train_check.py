###########################################################################
# Created by: Hang Zhang
# Email: zhang.hang@rutgers.edu
# Copyright (c) 2017
###########################################################################

import os, sys
BASE_DIR = os.path.dirname(os.path.dirname(os.getcwd()))
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
import torchvision.transforms as transforms
from torch.nn.parallel.scatter_gather import gather

import encoding.utils as utils
from encoding.nn import SegmentationLosses
from encoding.datasets import *
from encoding.models import get_segmentation_model

BASE_DIR = '.'
CONFIG_PATH = 'sun_experiments/irb_psk_pdl_wt1/results/config.yaml'
SMY_PATH = os.path.dirname(CONFIG_PATH)
image_w, image_h = 640, 480

s = 'hf_000a'
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

load_kwargs = {'num_workers': args.workers, 'pin_memory': True} if args.cuda else {}
trainset = SUNRGBD(transform=transforms.Compose([scaleNorm(),
                                                   RandomScale((1.0, 1.4)),
                                                   RandomHSV((0.9, 1.1),(0.9, 1.1),(25, 25)),
                                                   RandomCrop(image_h, image_w),
                                                   RandomFlip(),
                                                   ToTensor(),
                                                   Normalize()]),
                     phase_train=True)
trainloader = data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, **load_kwargs)   # drop_last=True pin_mem

testset = SUNRGBD(transform=transforms.Compose([scaleNorm(), ToTensor(), Normalize()]),phase_train=False)
    # scaleNorm(),  #不需要， recale
    # RandomScale((1.0, 1.4)),
    # RandomHSV((0.9, 1.1), (0.9, 1.1), (25, 25)),
    # RandomCrop(image_h, image_w),
    # RandomFlip(),

valloader = data.DataLoader(testset, batch_size=args.batch_size, drop_last=False, shuffle=False, **load_kwargs)  # drop_last=True pin_mem
nclass = trainset.NUM_CLASS

weight = trainset.compute_class_weights('median_frequency')


# model
model = get_segmentation_model(args.model, dataset=args.dataset, backbone=args.backbone, pretrained=True, dtype = args.dtype,
                               root='./encoding/models/pretrain', **model_kwargs)

print(model)

# optimizer using different LR
base_modules = [model.base, model.d_layer1, model.d_layer2, model.d_layer3, model.d_layer4]
base_ids = utils.get_param_ids(base_modules)
base_params = filter(lambda p: id(p) in base_ids, model.parameters())
other_params = filter(lambda p: id(p) not in base_ids, model.parameters())
optimizer = torch.optim.SGD([{'params': base_params, 'lr': args.lr},
                             {'params': other_params, 'lr': args.lr*10}],
                            lr=args.lr,momentum=args.momentum, weight_decay=args.weight_decay)

# criterions

type = None
if train_args['class_weight'] is not None:
    import pickle
    wt_path = os.path.join(trainset.BASE_DIR, 'weight', '{}.pickle'.format(train_args['class_weight']) )
    with open(wt_path, 'rb') as handle:
        wt = pickle.load(handle)
    class_wt = torch.FloatTensor(wt)  # .to(self.device)
else:
    class_wt = None

type = None
criterion = SegmentationLosses(aux=model_kwargs.get('aux'),
                                    nclass=nclass, weight=class_wt,
                                    aux_weight=train_args['aux_weight'], type=type)


scheduler = utils.LR_Scheduler_Head(args.lr_scheduler, args.lr, args.epochs, len(trainloader), warmup_epochs=1)
best_pred = 0.0

# using cuda
device = torch.device("cuda:0" if args.cuda else "cpu")

model = model.to(device)




# ==================== train =====================
train_loss = 0.0
epoch = 1
model.train()
for i, d  in enumerate(trainloader):
    print('1 batch')
    break

image, dep, target = d['image'], d['depth'], d['label'].long()

scheduler(optimizer, i, epoch, best_pred)

optimizer.zero_grad()



outputs = model(image, dep)
loss = criterion(*outputs, target)
loss.backward()
optimizer.step()

train_loss += loss.item()


# ==================== eval =====================

for i, dt in enumerate(valloader):
    image, dep, target = dt['image'], dt['depth'], dt['label'].long()
    break


model.eval()
pred = model(image, dep)
loss = criterion(*pred, target)
pred = pred[0]
correct, labeled = utils.batch_pix_accuracy(pred.data, target)
inter, union = utils.batch_intersection_union(pred.data, target, nclass)
