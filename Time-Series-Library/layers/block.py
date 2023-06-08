import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_, DropPath
from layers.Transformer_EncDec import Decoder, DecoderLayer, Encoder, EncoderLayer, ConvLayer
from layers.SelfAttention_Family import ProbAttention, AttentionLayer, DSAttention, FullAttention


class GRN(nn.Module):
    """ GRN (Global Response Normalization) layer
    """

    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, 1, 1, dim))
        self.beta = nn.Parameter(torch.zeros(1, 1, 1, dim))

    def forward(self, x):
        # L2正则：主要用来防止模型过拟合，直观上理解就是L2正则化是对于大数值的权重向量进行严厉惩罚。鼓励参数是较小值
        Gx = torch.norm(x, p=2, dim=(1, 2), keepdim=True)
        Nx = Gx / (Gx.mean(dim=-1, keepdim=True) + 1e-6)
        return self.gamma * (x * Nx) + self.beta + x


class GRN_1d(nn.Module):
    """ GRN (Global Response Normalization) layer
    """

    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, 1, dim))
        self.beta = nn.Parameter(torch.zeros(1, 1, dim))

    def forward(self, x):
        # L2正则：主要用来防止模型过拟合，直观上理解就是L2正则化是对于大数值的权重向量进行严厉惩罚。鼓励参数是较小值
        Gx = torch.norm(x, p=2, dim=(1), keepdim=True)
        Nx = Gx / (Gx.mean(dim=-1, keepdim=True) + 1e-6)
        return self.gamma * (x * Nx) + self.beta + x


class Block(nn.Module):
    """ ConvNeXtV2 Block.

    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
    """

    def __init__(self, dim, drop_path=0.):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)  # depthwise conv
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)  # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.grn = GRN(4 * dim)
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.grn(x)
        x = self.pwconv2(x)
        x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)

        x = input + self.drop_path(x)
        return x


# block1d的输入和输出都是[batch_size,num_channel,seq_len]这种类似于图像的写法
class Block_1d(nn.Module):
    """ ConvNeXtV2 Block.

    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
    """

    def __init__(self, dim, drop_path=0.):
        super().__init__()
        self.dwconv = nn.Conv1d(dim, dim, kernel_size=7, padding=3, groups=dim)  # depthwise conv
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)  # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.grn = GRN_1d(4 * dim)
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 1)  # (N, C, S) -> (N, S, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.grn(x)
        x = self.pwconv2(x)
        x = x.permute(0, 2, 1)  # (N, S, C) -> (N, C, S)

        x = input + self.drop_path(x)
        return x


# 创建一个包含 permute 操作的自定义层
class PermuteLayer(nn.Module):
    def __init__(self, dim_order):
        super(PermuteLayer, self).__init__()
        self.dim_order = dim_order

    def forward(self, x):
        return x.permute(self.dim_order)


"""
input:【batch_size,seq_len,dim]
output:[batch_size,seq_len,dim]
对于本例input和output都是d_model
input dim:就是dim
hidden_dim:输出线性层中的尺寸,也是lstm中d_model的尺寸
layer_dim:lstm层数
output_dim:分类的个数
"""


class lstm_n(nn.Module):
    """Very simple implementation of LSTM-based time-series classifier."""

    # num_layers – Number of recurrent layers. E.g., setting num_layers=2 would mean stacking two LSTMs
    # batch_first – If True, then the input and output tensors are provided as (batch, seq, feature) instead of (seq, batch, feature).
    def __init__(self, configs=None):
        super().__init__()
        self.input_dim = configs.d_model
        self.hidden_dim = configs.d_model
        self.layer_dim = configs.e_layers

        # 注意这里设置了batch_first所以第一个维度是batch，lstm第二个input是输出的维度，第三个是lstm的层数
        self.lstm = nn.LSTM(self.input_dim, self.hidden_dim, self.layer_dim, batch_first=True)

    def forward(self, x):
        # init_hidden并不是魔法函数，是每次循环时候手动执行更新的
        h0, c0 = self.init_hidden(x)
        out, (hn, cn) = self.lstm(x, (h0, c0))
        # (N, L, D * H_{out})(N,L,D∗H_out) D代表的是direction，如果是双向lstm的话则d为2 else 1，L代表的是sequence
        return out

    def init_hidden(self, x):
        # (lstm层的个数，batch_size,输出层的个数)
        h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim)
        c0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim)
        return [t.cuda() for t in (h0, c0)]


class fusion_layer(nn.Module):
    def __init__(self, configs, cat_mode='none', attn='prob'):
        super(fusion_layer, self).__init__()
        self.cat_mode = cat_mode
        if self.cat_mode == 'dim':
            # concate on d_model dim
            self.linear = nn.Linear(configs.d_model * 2, configs.d_model)
        elif self.cat_mode == 'seq':
            self.linear = nn.Linear(configs.seq_len * 2, configs.seq_len)
        elif self.cat_mode == 'former':
            # Decoder
            attn = 'prob'
            if attn == 'full':
                Attn = FullAttention
            elif attn == 'prob':
                Attn = ProbAttention
            elif attn == 'dsa':
                Attn = DSAttention

            self.decoder = Decoder(
                [
                    DecoderLayer(
                        AttentionLayer(
                            Attn(True, configs.factor, attention_dropout=configs.dropout,
                                 output_attention=False),
                            configs.d_model, configs.n_heads),
                        AttentionLayer(
                            Attn(False, configs.factor, attention_dropout=configs.dropout,
                                 output_attention=False),
                            configs.d_model, configs.n_heads),
                        configs.d_model,
                        configs.d_ff,
                        dropout=configs.dropout,
                        activation=configs.activation,
                    )
                    for l in range(configs.d_layers)
                ],
                norm_layer=torch.nn.LayerNorm(configs.d_model),

            )

    def forward(self, x1, x2):
        if self.cat_mode == 'dim':
            out = torch.cat((x1, x2), dim=2)
            out = self.linear(out)
        elif self.cat_mode == 'seq':
            out = torch.cat((x1, x2), dim=1)
            out = out.permute(0, 2, 1)
            out = self.linear(out)
            out = out.permute(0, 2, 1)
        elif self.cat_mode == 'none':
            out = x1 + x2
        elif self.cat_mode == 'former':
            out = self.decoder(x1, x2, x_mask=None, cross_mask=None)
        return out
