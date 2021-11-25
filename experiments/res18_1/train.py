###########################################################################
# Created by: Hang Zhang
# Email: zhang.hang@rutgers.edu
# Copyright (c) 2017
###########################################################################
import os, sys
BASE_DIR = os.path.dirname(os.path.dirname(os.getcwd()))
sys.path.append(BASE_DIR)
import yaml
import numpy as np
from addict import Dict

import torch
import torch.nn as nn
from torch.utils import data
from tensorboardX import SummaryWriter
import torchvision.transforms as transform

import encoding.utils as utils
from encoding.nn import SegmentationLosses, SegmentationAuxLosses
from encoding.datasets import get_dataset
from encoding.models import get_segmentation_model
CONFIG_PATH = './results/config.yaml'
SMY_PATH = os.path.dirname(CONFIG_PATH)
GPUS = [0, 1]

# model settings
print('[Exp Name]:', sys.argv[1])
print("-------mark program start----------")
# configuration
model_kwargs = utils.get_model_args(sys.argv[1])
model_kwargs = {k:v for k, v in model_kwargs.items() if v is not None}
print('++++++++++++++++++++model_kwargs {}++++++++++++++++++++'.format(model_kwargs))

train_setting = utils.split_train_args(sys.argv[1])   # 用来定义output的路径
train_args = utils.get_train_args(train_setting)      # 用来修改模型训练的超参
print('++++++++++++++++++++train_args {}++++++++++++++++++++'.format(train_args))



class Trainer():
    def __init__(self, args):
        self.args = args
        # data transforms
        input_transform = transform.Compose([
            transform.ToTensor(),  # convert RGB [0,255] to FloatTensor in range [0, 1]
            transform.Normalize([.485, .456, .406], [.229, .224, .225])])  # mean and std based on imageNet
        dep_transform = transform.Compose([
            transform.ToTensor(),
            transform.Normalize(mean=[0.2798], std=[0.1387])  # mean and std for depth
        ])
        # dataset
        data_kwargs = {'transform': input_transform, 'dep_transform': dep_transform,
                       'base_size': args.base_size, 'crop_size': args.crop_size}
        trainset = get_dataset(args.dataset, split=args.train_split, mode='train', **data_kwargs)
        testset = get_dataset(args.dataset, split='val', mode='val', **data_kwargs)
        # dataloader
        kwargs = {'num_workers': args.workers, 'pin_memory': True} if args.cuda else {}
        self.trainloader = data.DataLoader(trainset, batch_size=args.batch_size, drop_last=True, shuffle=True, **kwargs)
        self.valloader = data.DataLoader(testset, batch_size=args.batch_size, drop_last=False, shuffle=False, **kwargs)
        self.nclass = trainset.num_class

        # model and params
        model = get_segmentation_model(args.model, dataset=args.dataset, backbone=args.backbone, pretrained=True,
                                       root='../../encoding/models/pretrain', dtype=args.dtype,
                                       **model_kwargs)
        print(model)

        # using cuda
        self.device = torch.device("cuda:0" if args.cuda else "cpu")
        self.model = model.to(self.device)

        # optimizer using different LR
        base_modules = [model.base, model.d_layer0, model.d_layer1, model.d_layer2, model.d_layer3, model.d_layer4]
        base_ids = utils.get_param_ids(base_modules)
        base_params = filter(lambda p: id(p) in base_ids, model.parameters())
        other_params = filter(lambda p: id(p) not in base_ids, model.parameters())
        self.optimizer = torch.optim.SGD([{'params': base_params, 'lr': args.lr},
                                          {'params': other_params, 'lr': args.lr * 10}],
                                         lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

        # criterions
        if train_args['class_weight'] is not None:
            import pickle
            wt_path = os.path.join(trainset.BASE_DIR, 'weight', '{}.pickle'.format(train_args['class_weight']) )
            with open(wt_path, 'rb') as handle:
                wt = pickle.load(handle)
            class_wt = torch.FloatTensor(wt).to(self.device)
        else:
            class_wt = None

        self.criterion = SegmentationLosses(aux=model_kwargs.get('aux'),
                                            nclass=self.nclass, weight=class_wt,
                                            aux_weight=args.aux_weight, type=None)


        # lr scheduler
        self.scheduler = utils.LR_Scheduler_Head(args.lr_scheduler, args.lr, args.epochs,
                                                 iters_per_epoch=len(self.trainloader), warmup_epochs=5)
        self.best_pred = 0.0

        # for writing summary
        path = "/".join(("{}-{}".format(*i) for i in model_kwargs.items()))
        if train_setting is not None:
            path = os.path.join(path, train_setting)
        self.writer = SummaryWriter(os.path.join(SMY_PATH, path))
        # resuming checkpoint
        if args.resume is not None and args.resume != 'None':
            if not os.path.isfile(args.resume):
                raise RuntimeError("=> no checkpoint found at '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            if self.multi_gpu:
                self.model.module.load_state_dict(checkpoint['state_dict'])
            else:
                self.model.load_state_dict(checkpoint['state_dict'])
            if not args.ft:
                self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.best_pred = checkpoint['best_pred']
            print("=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))
        # clear start epoch if fine-tuning
        if args.ft:
            args.start_epoch = 0

    def training(self, epoch):
        train_loss = 0.0
        self.model.train()

        total_inter, total_union, total_correct, total_label, total_loss = 0, 0, 0, 0, 0
        for i, (image, dep, target) in enumerate(self.trainloader):
            image, dep, target = image.to(self.device), dep.to(self.device), target.to(self.device)
            self.scheduler(self.optimizer, i, epoch, self.best_pred)
            self.optimizer.zero_grad()
            outputs = self.model(image, dep)

            loss = self.criterion(*outputs, target)

            loss.backward()
            self.optimizer.step()

            correct, labeled = utils.batch_pix_accuracy(outputs[0].data, target)
            inter, union = utils.batch_intersection_union(outputs[0].data, target, self.nclass)
            total_correct += correct
            total_label += labeled
            total_inter += inter
            total_union += union
            train_loss += loss.item()

            if (i+1) % 50 == 0:
                print('epoch {}, step {}, loss {}'.format(epoch + 1, i + 1, train_loss / 50))
                self.writer.add_scalar('train_loss', train_loss / 50, epoch * len(self.trainloader) + i)
                train_loss = 0.0
        pixAcc = 1.0 * total_correct / (np.spacing(1) + total_label)
        IOU = 1.0 * total_inter / (np.spacing(1) + total_union)
        mIOU = IOU.mean()
        print('epoch {}, pixel Acc {}, mean IOU {}'.format(epoch + 1, pixAcc, mIOU))
        self.writer.add_scalar("mean_iou/train", mIOU, epoch)
        self.writer.add_scalar("pixel accuracy/train", pixAcc, epoch)
        self.writer.add_scalar('check_info/base_lr0', self.optimizer.param_groups[0]['lr'], epoch)
        self.writer.add_scalar('check_info/other_lr1', self.optimizer.param_groups[1]['lr'], epoch)

    def train_n_evaluate(self):

        for epoch in range(self.args.epochs):
            # run on one epoch
            print("\n===============train epoch {}/{} ==========================\n".format(epoch, self.args.epochs))

            # one full pass over the train set
            self.training(epoch)

            # evaluate for one epoch on the validation set
            print('\n===============start testing, training epoch {}\n'.format(epoch))
            pixAcc, mIOU, loss = self.validation(epoch)
            print('evaluation pixel acc {}, mean IOU {}, loss {}'.format(pixAcc, mIOU, loss))

            # save the best model
            is_best = False
            new_pred = (pixAcc + mIOU) / 2
            if new_pred > self.best_pred:
                is_best = True
                self.best_pred = new_pred
            path = 'runs/' + "/".join(("{}-{}".format(*i) for i in model_kwargs.items()))
            if train_setting is not None:
                path = os.path.join(path, train_setting)
            utils.save_checkpoint({'epoch': epoch + 1,
                                   'state_dict': self.model.state_dict(),
                                   'optimizer': self.optimizer.state_dict(),
                                   'best_pred': self.best_pred}, self.args, is_best, path = path)

    def validation(self, epoch):
        # Fast test during the training
        def eval_batch(model, image, dep, target):
            # model, image, target already moved to gpus
            pred = model(image, dep)
            loss = self.criterion(*pred, target)
            pred = pred[0]
            correct, labeled = utils.batch_pix_accuracy(pred.data, target)
            inter, union = utils.batch_intersection_union(pred.data, target, self.nclass)
            return correct, labeled, inter, union, loss

        self.model.eval()
        total_inter, total_union, total_correct, total_label, total_loss = 0, 0, 0, 0, 0
        for i, (image, dep, target) in enumerate(self.valloader):
            image, dep, target = image.to(self.device), dep.to(self.device), target.to(self.device)
            with torch.no_grad():
                correct, labeled, inter, union, loss = eval_batch(self.model, image, dep, target)

            total_correct += correct
            total_label += labeled
            total_inter += inter
            total_union += union
            total_loss += loss.item()
            pixAcc = 1.0 * total_correct / (np.spacing(1) + total_label)
            IOU = 1.0 * total_inter / (np.spacing(1) + total_union)
            mIOU = IOU.mean()

            if i % 40 == 0:
                print('eval mean IOU {}'.format(mIOU))

        loss = total_loss / len(self.valloader)
        self.writer.add_scalar("mean_iou/val", mIOU, epoch)
        self.writer.add_scalar("pixel accuracy/val", pixAcc, epoch)

        return pixAcc, mIOU, loss


if __name__ == "__main__":
    print("-------mark program start----------")
    # configuration
    args = Dict(yaml.safe_load(open(CONFIG_PATH)))
    args.cuda = (args.use_cuda and torch.cuda.is_available())
    args.resume = None if args.resume=='None' else args.resume
    torch.manual_seed(args.seed)

    for k, v in train_args.items():
        if v is not None:
            args[k] = v
            print('args {} is set to {}'.format(k, args[k]))

    trainer = Trainer(args)
    # import pdb; pdb.set_trace()
    print('Starting Epoch:', trainer.args.start_epoch)
    print('Total Epoches:', trainer.args.epochs)
    trainer.train_n_evaluate()


