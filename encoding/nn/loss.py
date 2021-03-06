import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable

__all__ = ['LabelSmoothing', 'NLLMultiLabelSmooth', 'SegmentationLosses', 'SegmentationAuxLosses']

class LabelSmoothing(nn.Module):
    """
    NLL loss with label smoothing.
    """
    def __init__(self, smoothing=0.1):
        """
        Constructor for the LabelSmoothing module.
        :param smoothing: label smoothing factor
        """
        super(LabelSmoothing, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing

    def forward(self, x, target):
        logprobs = torch.nn.functional.log_softmax(x, dim=-1)

        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()

class NLLMultiLabelSmooth(nn.Module):
    def __init__(self, smoothing = 0.1):
        super(NLLMultiLabelSmooth, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing

    def forward(self, x, target):
        if self.training:
            x = x.float()
            target = target.float()
            logprobs = torch.nn.functional.log_softmax(x, dim = -1)
    
            nll_loss = -logprobs * target
            nll_loss = nll_loss.sum(-1)
    
            smooth_loss = -logprobs.mean(dim=-1)
    
            loss = self.confidence * nll_loss + self.smoothing * smooth_loss
    
            return loss.mean()
        else:
            return torch.nn.functional.cross_entropy(x, target)

class SegmentationLosses(nn.CrossEntropyLoss):
    """2D Cross Entropy Loss with Auxilary Loss"""
    def __init__(self, se_loss=False, se_weight=0.2, nclass=-1,
                 aux=None, aux_weight=0.4, weight=None, ignore_index=-1, type=None):
        super(SegmentationLosses, self).__init__(weight, None, ignore_index)
        self.se_loss = se_loss
        self.aux = aux
        self.type = type
        self.nclass = nclass
        self.se_weight = se_weight
        self.aux_weight = aux_weight
        self.bceloss = nn.BCELoss(weight)
        self.ceploss = nn.CrossEntropyLoss(ignore_index=-1)

    def forward(self, *inputs):
        if not self.se_loss and not self.aux:
            return super(SegmentationLosses, self).forward(*inputs)
        elif not self.se_loss:
            target = inputs[-1]
            loss1 = super(SegmentationLosses, self).forward(inputs[0], target)
            #loss0 = loss
            aux_loss = []
            for i in range(1, len(self.aux)+1):
                aux_feat = inputs[i]
                _, _, h, w = aux_feat.size()
                aux_target = F.interpolate(target.unsqueeze(1).float(), size=(h, w)).long().squeeze(1)
                if self.type is None:
                    aux_loss.append( super(SegmentationLosses, self).forward(aux_feat, aux_target) )
                elif self.type == 's':
                    aux_loss.append( self.ceploss(aux_feat, aux_target) )
            loss2 = sum(aux_loss)/len(aux_loss)
            return loss1 + loss2*self.aux_weight

        elif not self.aux:
            pred, se_pred, target = tuple(inputs)
            se_target = self._get_batch_label_vector(target, nclass=self.nclass).type_as(pred)
            loss1 = super(SegmentationLosses, self).forward(pred, target)
            loss2 = self.bceloss(torch.sigmoid(se_pred), se_target)
            return loss1 + self.se_weight * loss2
        else:
            pred1, se_pred, pred2, target = tuple(inputs)
            se_target = self._get_batch_label_vector(target, nclass=self.nclass).type_as(pred1)
            loss1 = super(SegmentationLosses, self).forward(pred1, target)
            loss2 = super(SegmentationLosses, self).forward(pred2, target)
            loss3 = self.bceloss(torch.sigmoid(se_pred), se_target)
            return loss1 + self.aux_weight * loss2 + self.se_weight * loss3

    @staticmethod
    def _get_batch_label_vector(target, nclass):
        # target is a 3D Variable BxHxW, output is 2D BxnClass
        batch = target.size(0)
        tvect = Variable(torch.zeros(batch, nclass))
        for i in range(batch):
            hist = torch.histc(target[i].cpu().data.float(), 
                               bins=nclass, min=0,
                               max=nclass-1)
            vect = hist>0
            tvect[i] = vect
        return tvect


class SegmentationAuxLosses(nn.CrossEntropyLoss):
    """2D Cross Entropy Loss with Auxilary Loss"""
    def __init__(self, nclass=-1, aux_weight=0.4, weight=None, ignore_index=-1):
        super(SegmentationAuxLosses, self).__init__(weight, None, ignore_index)
        self.nclass = nclass
        self.aux_weight = aux_weight
        self.bceloss = nn.BCELoss(weight)

    def forward(self, *inputs):
        target = inputs[-1]
        aux_loss = []
        for i in range(1, len(self.aux)+1):
            aux_feat = inputs[i]
            _, _, h, w = aux_feat.size()
            aux_target = F.interpolate(target.unsqueeze(1).float(), size=(h, w)).long().squeeze(1)
            aux_loss.append( super(SegmentationLosses, self).forward(aux_feat, aux_target) )
        loss2 = sum(aux_loss)/len(aux_loss)
        return loss2*self.aux_weight
