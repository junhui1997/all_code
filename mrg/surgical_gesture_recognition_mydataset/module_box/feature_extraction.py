import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import torchvision

# cnn到这一层输出的channel数目就是512，这是不会变得，所以到下一层的token_learner也没啥影响
# out x[512,a,b] a,b随图像的尺寸变化
class cnn_feature(nn.Module):
    def __init__(self):
        super().__init__()
        # 需要用resnet50的话直接改成50jike
        self.resnet = timm.create_model('resnet18', pretrained=True)
        #self.resnet = torchvision.models.resnet18(pretrained = True)
        self.resnet.eval()
        self.resnet_list = list(self.resnet.children())
        # dprint('len of resnet',len(self.resnet_list))

    def forward(self, x):
        # 取resnet结果作为下一层的输入,倒数第二层那里很奇怪，我就直接给flatten了
        with torch.no_grad():
            for i in range(len(self.resnet_list) - 2):
                x = self.resnet_list[i](x)
                # dprint(i,x.shape)
        return x


