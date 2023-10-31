import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from models import Autoformer, Transformer, TimesNet, Nonstationary_Transformer, DLinear, FEDformer, \
    Informer, LightTS, Reformer, ETSformer, Pyraformer, PatchTST, MICN, Crossformer, lstm, conv_net, bp, lstm_fcn, fcn, fcn_m, convnext1d, my, sinc_net, consinc_net

'''
(batch_size,seq_len,d_model)->(batch_size,c_out)
'''


class linear_proj(nn.Module):
    def __init__(self, configs):
        super(linear_proj, self).__init__()
        self.act = F.gelu
        self.dropout = nn.Dropout(configs.dropout)
        self.projection = nn.Linear(configs.d_model * configs.seq_len, configs.d_model)
        self.projection2 = nn.Linear(configs.d_model, configs.c_out)
        init.xavier_uniform_(self.projection.weight)
        init.xavier_uniform_(self.projection2.weight)

    def forward(self, x):
        out = self.act(x)
        out = self.dropout(out)
        out = out.reshape(out.shape[0], -1)
        out = self.projection(out)
        out = self.projection2(out)
        return out


class neural_pd(nn.Module):
    def __init__(self, encoder, configs):
        super(neural_pd, self).__init__()
        self.encoder = encoder
        self.batch_size = configs.batch_size
        self.seq_len = configs.seq_len
        self.linear_proj = linear_proj(configs)

    def forward(self, x):
        x_mark = torch.ones((x.shape[0], self.seq_len)).to(x.device)
        out = self.encoder(x, x_mark, None, None)
        out = self.linear_proj(out)
        return out
