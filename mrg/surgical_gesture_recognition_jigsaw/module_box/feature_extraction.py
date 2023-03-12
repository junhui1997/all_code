import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import torchvision

# cnn到这一层输出的channel数目就是512，这是不会变得，所以到下一层的token_learner也没啥影响
# out x[512,a,b] a,b随图像的尺寸变化
class cnn_feature(nn.Module):
    def __init__(self, feature_type='conv'):
        super().__init__()
        # 需要用resnet50的话直接改成50jike
        self.feature_type = feature_type
        self.resnet = timm.create_model('resnet18', pretrained=True)
        #self.resnet = torchvision.models.resnet18(pretrained = True)
        self.resnet.eval()
        self.resnet_list = list(self.resnet.children())
        # dprint('len of resnet',len(self.resnet_list))

    def forward(self, x):
        # 取resnet结果作为下一层的输入,倒数第二层那里很奇怪，我就直接给flatten了
        if self.feature_type == 'conv':
            factor = 2
        else:
            factor = 0
        with torch.no_grad():
            for i in range(len(self.resnet_list) - factor):
                x = self.resnet_list[i](x)
                # dprint(i,x.shape)
        return x


class cnn_feature50(nn.Module):
    def __init__(self, feature_type='conv'):
        super().__init__()
        # 需要用resnet50的话直接改成50jike
        self.feature_type = feature_type
        self.resnet = timm.create_model('resnet50', pretrained=True)
        #self.resnet = torchvision.models.resnet18(pretrained = True)
        self.resnet.eval()
        self.resnet_list = list(self.resnet.children())
        self.res_fc = nn.Linear(2048, 512)
        # dprint('len of resnet',len(self.resnet_list))

    def forward(self, x):
        # 取resnet结果作为下一层的输入,倒数第二层那里很奇怪，我就直接给flatten了
        if self.feature_type == 'conv':
            factor = 2
        else:
            factor = 0
        with torch.no_grad():
            for i in range(len(self.resnet_list) - factor):
                x = self.resnet_list[i](x)
                # dprint(i,x.shape)
        if self.feature_type == 'conv':
            x = x.permute(0, 3, 2, 1)
            x = self.res_fc(x)
            x = x.permute(0, 3, 2, 1)
        else:
            pass
        return x

