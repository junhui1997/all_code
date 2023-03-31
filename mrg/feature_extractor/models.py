import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import torchvision


class cnn_feature18(nn.Module):
    def __init__(self, args, dropout=0.5):
        super().__init__()
        self.dropout = dropout
        self.hidden_state = 128
        self.resnet = timm.create_model('resnet18', pretrained=True)
        self.resnet_list = list(self.resnet.children())
        # dprint('len of resnet',len(self.resnet_list))
        self.fc_h = nn.Linear(self.feature_dim, self.hidden_state)
        self.fc = nn.Linear(self.hidden_state, args.num_classes)
        nn.init.xavier_uniform_(self.fc_h.weight)
        nn.init.xavier_uniform_(self.fc.weight)

    def forward(self, x, return_feature=False ):
        for i in range(len(self.resnet_list)):
            x = self.resnet_list[i](x)
        out_hidden = self.fc_h(x)

        out_hidden_2 = F.relu(out_hidden)

        out_hidden_2 = F.dropout(out_hidden_2, p=self.dropout)

        out = self.fc(out_hidden_2)

        if return_feature:
            return out, out_hidden
        else:
            return out
