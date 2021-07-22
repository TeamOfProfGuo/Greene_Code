import os
import torch
import torch.nn as nn
import torchvision.models as models

__all__ = ['get_resnet18']

def get_resnet18(pretrained=True, input_dim = 3, f_path='./../../encoding/models/pretrain/resnet18-5c106cde.pth'):
    assert input_dim in (1, 3, 4)
    model = models.resnet18(pretrained=False)

    if pretrained:
        # Check weights file
        if not os.path.exists(f_path):
            raise FileNotFoundError('The pretrained model cannot be found.')
        model.load_state_dict(torch.load(f_path), strict=False)

        if input_dim != 3:
            model.conv1 = nn.Conv2d(input_dim, 64, kernel_size=7, stride=2, padding=3, bias=False)
    else:
        raise ValueError('Please use pretrained resnet18.')
    
    return model