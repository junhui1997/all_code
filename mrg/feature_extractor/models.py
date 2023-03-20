import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import torchvision


class cnn_feature18(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.resnet = timm.create_model('resnet18', pretrained=True)
        self.resnet_list = list(self.resnet.children())
        # dprint('len of resnet',len(self.resnet_list))
        self.fc_h = nn.Linear(1000, 128)
        self.fc = nn.Linear(128, args.num_classes)

    def forward(self, x):
        for i in range(len(self.resnet_list)):
            x = self.resnet_list[i](x)
        x = self.fc_h(x)
        x = self.fc(x)
        return x
