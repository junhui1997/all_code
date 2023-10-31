import torch
import torch.nn as nn
from models import Autoformer, Transformer, TimesNet, Nonstationary_Transformer, DLinear, FEDformer, \
    Informer, LightTS, Reformer, ETSformer, Pyraformer, PatchTST, MICN, Crossformer, lstm, conv_net, bp, lstm_fcn, fcn, fcn_m, convnext1d, my, sinc_net, consinc_net
class neural_pd(nn.Module):
    def __init__(self, encoder, configs):
        super(neural_pd, self).__init__()
        self.encoder = encoder
        self.batch_size = configs.batch_size
        self.seq_len = configs.seq_len

    def forward(self, x):
        x_mark = torch.ones((self.batch_size, self.seq_len)).to(x.device)
        out = self.encoder(x, x_mark, None, None)
        return out
