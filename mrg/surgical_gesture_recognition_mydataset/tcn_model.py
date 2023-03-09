from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from GNN.rgcn import RGCN
from GNN.gcn import GCN
from dgl import DGLGraph
import numpy as np
import torch.nn.functional as F

from models.encoder import Encoder, EncoderLayer, ConvLayer
from models.encoder import Encoder as att_Encoder
from models.attn import FullAttention, ProbAttention, AttentionLayer
from models.embed import DataEmbedding
from module_box.feature_extraction import cnn_feature, cnn_feature50
from module_box.token_learner import token_learner

import time




class TcnGcnNet(nn.Module):
    def __init__(self, class_num, hidden_state,
                 encoder_params,
                 decoder_params,
                 mid_lstm_params):

        super(TcnGcnNet, self).__init__()

        self.s = 8
        self.cnn_feature = cnn_feature50()
        self.token_learner = token_learner(S=self.s)

        d_model = 512
        attn = 'prob'
        factor = 5
        n_heads = 8
        dropout = 0.0
        d_ff = 512
        activation = 'gelu'
        e_layers = 10
        distil = None
        embed = 'fixed'
        # 192 是最后一个维度，也就是dim
        self.enc_embedding = DataEmbedding(512, d_model, embed, dropout)
        Attn = ProbAttention if attn == 'prob' else FullAttention
        self.encoder_all = att_Encoder(
            [
                EncoderLayer(
                    AttentionLayer(Attn(False, factor, attention_dropout=dropout, output_attention=False),
                                   d_model, n_heads, mix=False),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation
                ) for l in range(e_layers)
            ],
            [
                ConvLayer(
                    d_model
                ) for l in range(e_layers - 1)
            ] if distil else None,
            norm_layer=torch.nn.LayerNorm(d_model)
        )

        attn = 'full'
        e_layers = 5
        self.kine_enc_embedding = DataEmbedding(14, d_model, embed, dropout)
        Attn = ProbAttention if attn == 'prob' else FullAttention
        self.kine_encoder = att_Encoder(
            [
                EncoderLayer(
                    AttentionLayer(Attn(False, factor, attention_dropout=dropout, output_attention=False),
                                   d_model, n_heads, mix=False),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation
                ) for l in range(e_layers)
            ],
            [
                ConvLayer(
                    d_model
                ) for l in range(e_layers - 1)
            ] if distil else None,
            norm_layer=torch.nn.LayerNorm(d_model)
        )
        # 512是d_model,128是左右两边的128,5是num_classes
        self.vision_conv = ConvLayer(d_model, self.s)
        self.vision_fc = torch.nn.Linear(512, 256)
        self.kine_fc = torch.nn.Linear(512, 256)
        self.new_fc = torch.nn.Linear(512, 5)


    #forward only change kinetic embedding
    def forward(self, x_vision, x_kinematics, return_emb=False):
        x_kine = self.kine_enc_embedding(x_kinematics)
        x_kine, attn_kine = self.kine_encoder(x_kine, attn_mask=None)

        batch_size, seq_len, _, _, _ = x_vision.shape
        # !!!!!非继承的tensor切记要移动到cuda中去
        x_visions = torch.Tensor(batch_size, seq_len * self.s, 512).cuda()
        for i in range(seq_len):
            # x_feature在token learner之后是[batch_size,self.s,512]
            x_vision_single = self.cnn_feature(x_vision[:, i, :, :, :])
            x_vision_single = self.token_learner(x_vision_single)
            x_visions[:, i * self.s:(i + 1) * self.s, :] = x_vision_single
        x_visions = self.vision_conv(x_visions)
        # 最终需要使用的结果是x_left,x_right,x_vision:[batch_size,video_len,64]

        x_kine = self.kine_fc(x_kine)
        x_visions = self.vision_fc(x_visions)
        x_fu = torch.cat((x_kine, x_visions), dim=2)
        x_fu = self.enc_embedding(x_fu)
        x_fu, attns = self.encoder_all(x_fu, attn_mask=None)
        out = self.new_fc(x_fu)

        if return_emb:
            return out, 0
        else:
            return out

    #forward only change final graph network
   # def forward(self, x_vision, x_kinematics, return_emb=False):
   #      x_left = x_kinematics[:, :, :7]
   #      x_right = x_kinematics[:, :, 7:]
   #
   #      x_vision = self.tcn_vision(x_vision)
   #      x_left_t = self.tcn_left(x_left)
   #      x_right_t = self.tcn_right(x_right)
   #
   #      x_left_l, _ = self.lstm_left(x_left)
   #      x_right_l, _ = self.lstm_right(x_right)
   #
   #      x_left = (x_left_t + x_left_l) / 2
   #      x_right = (x_right_t + x_right_l) / 2
   #      # 最终需要使用的结果是x_left,x_right,x_vision:[batch_size,video_len,64]
   #
   #      x_fu = torch.cat((x_left, x_right, x_vision), dim=2)
   #      x_fu = self.enc_embedding(x_fu)
   #      x_fu, attns = self.encoder_all(x_fu, attn_mask=None)
   #      out = self.new_fc(x_fu)
   #
   #      if return_emb:
   #          return out, 0
   #      else:
   #          return out