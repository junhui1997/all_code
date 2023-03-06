import torch
import torch.nn as nn
import torch.nn.functional as F

import math
"""
    return [1,seq_len,d_model]
"""
class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False
        # position.shape是[5000,1]
        position = torch.arange(0, max_len).float().unsqueeze(1)
        #div_term.shape = [256]
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        #这里时候是[5000,512],这里执行了一个广播机制
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # pe最后shape是[1,5000,512]
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x.shape[32,96,7]就是在dataloader里面输出的那个[batch_size,seq_len,7],7是feature的数目
        # 这里打印出来的输出值，也就是return的值的shape是[1,96,512],其实就是对上面那个pe的数值进行了裁剪
        # test = self.pe[:, :x.size(1)]
        return self.pe[:, :x.size(1)]

"""
    c_in:7不确定这是不是输入的通道的数目
    将输入的通道数目转化为了，d_model的数目
    输出结果是[batch_size,seq_len,d_model]
"""
class TokenEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(TokenEmbedding, self).__init__()
        padding = 1 if torch.__version__>='1.5.0' else 2
        # 这里只是1维卷积而不是1*1卷积，kernel=3，padding=1这样卷积完之后长度是不变的，因为kernel导致-2，padding左右各+1
        self.tokenConv = nn.Conv1d(in_channels=c_in, out_channels=d_model, 
                                    kernel_size=3, padding=padding, padding_mode='circular')
        # 对层进行了相应的初始化，token都是需要嵌入的
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight,mode='fan_in',nonlinearity='leaky_relu')

    def forward(self, x):
        # 从原本的x[batch_size,seq_len,c_in]变为了卷积结束的[batch_size,seq_len,d_model]
        test = x
        # 这里我理解的形状并不改变，因为permute之后又transpose了过来
        x = self.tokenConv(x.permute(0, 2, 1)).transpose(1,2)
        return x


class NoTimeEmbedding(nn.Module):
    def __init__(self, d_model):
        super(NoTimeEmbedding, self).__init__()
        self.d_model = d_model

    def forward(self, x):
        batch_size, seq_len, time_featue = x.shape
        # 这里因为不是从x继承过来的，所以需要cuda
        zero = torch.zeros(batch_size, seq_len, self.d_model).cuda()
        return zero

class DataEmbedding(nn.Module):
    def __init__(self, c_in, d_model, embed_type='None', dropout=0.1):
        super(DataEmbedding, self).__init__()

        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        # embedding是对三个量都分别嵌入后进行相加，之后需要对batch_size方向执行了增广操作
        x = self.value_embedding(x) + self.position_embedding(x)
        
        return self.dropout(x)