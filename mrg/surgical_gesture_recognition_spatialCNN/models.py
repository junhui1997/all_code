# Copyright (C) 2019  National Center of Tumor Diseases (NCT) Dresden, Division of Translational Surgical Oncology

import numpy as np

import torch
import torchvision
from torch import nn
from torch.nn.init import normal, constant

from tcn import TemporalConvNet
from dgl import DGLGraph
import torch.nn.functional as F

from transforms import GroupMultiScaleCrop, GroupRandomHorizontalFlip
from train_opts import num_cls_Kinetics


# ========ResNet50-LSTM-GNN===========
# TODO
class GestureClassifier(nn.Module):
    def __init__(self, num_class, dropout=0.5, snippet_length=1,
                 input_size=224):
        super(GestureClassifier, self).__init__()

        self._init_base_model(num_class, dropout, snippet_length,
                              input_size)

    def _init_base_model(self, num_class, dropout=0.5, snippet_length=1,
                         input_size=224):

        self.snippet_length = snippet_length
        self.input_size = input_size
        self.input_mean = [0.485, 0.456, 0.406]
        self.input_std = [0.229, 0.224, 0.225]
        self.dropout = dropout
        self.hidden_state = 128

        # TODO
        resnet = torchvision.models.resnet18(pretrained=True)
        last_layer_name = 'fc'
        self.feature_dim = getattr(resnet, last_layer_name).in_features # 512 #this is the flatten 2-10 flatten [batch_size,512]

        self.base_model = torch.nn.Sequential()
        self.base_model.add_module("conv1", resnet.conv1)
        self.base_model.add_module("bn1", resnet.bn1)
        self.base_model.add_module("relu", resnet.relu)
        self.base_model.add_module("maxpool", resnet.maxpool)
        self.base_model.add_module("layer1", resnet.layer1)
        self.base_model.add_module("layer2", resnet.layer2)
        self.base_model.add_module("layer3", resnet.layer3)
        self.base_model.add_module("layer4", resnet.layer4)
        self.base_model.add_module("avgpool", resnet.avgpool)

        self.fc_h = nn.Linear(self.feature_dim, self.hidden_state)
        self.fc = nn.Linear(self.hidden_state, num_class)
        nn.init.xavier_uniform_(self.fc_h.weight)
        nn.init.xavier_uniform_(self.fc.weight)

    def forward(self, input_img, return_feature=False):

        # image branch
        input_img = input_img.view(-1, 3, self.input_size, self.input_size)
        base_out = self.base_model(input_img)

        base_out = base_out.view(-1, self.feature_dim)

        out_hidden = self.fc_h(base_out)

        out_hidden_2 = F.relu(out_hidden)

        out_hidden_2 = F.dropout(out_hidden_2, p=self.dropout)

        out = self.fc(out_hidden_2)

        if return_feature:
            return out, out_hidden

        return out

    @property
    def crop_size(self):
        return self.input_size

    @property
    def scale_size(self):
        return self.input_size * 256 // 224

    @property
    def is_3D_architecture(self):
        return True

    def get_augmentation(self, crop_corners=True, do_horizontal_flip=True):
        if do_horizontal_flip:
            return torchvision.transforms.Compose([GroupMultiScaleCrop(self.input_size, [1, .875, .75, .66],
                                                                       fix_crop=crop_corners,
                                                                       more_fix_crop=crop_corners),
                                                   GroupRandomHorizontalFlip(is_flow=False)])

        else:
            return torchvision.transforms.Compose([GroupMultiScaleCrop(self.input_size, [1, .875, .75, .66],
                                                                       fix_crop=crop_corners,
                                                                       more_fix_crop=crop_corners)])

