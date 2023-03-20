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
from config import args
from models.encoder import Encoder, EncoderLayer, ConvLayer
from models.encoder import Encoder as att_Encoder
from models.attn import FullAttention, ProbAttention, AttentionLayer
from models.embed import DataEmbedding
from models.AutoCorrelation import AutoCorrelation, AutoCorrelationLayer
from models.Autoformer_EncDec import my_Layernorm, series_decomp
from module_box.feature_extraction import cnn_feature, cnn_feature50
import time


# This module should be tested carefully
class LSTM_Layer(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers,
                 bi_dir=True, use_gru=True):
        super(LSTM_Layer, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bi_dir = bi_dir
        self.use_gru = use_gru

        if self.use_gru:
            self.lstm = nn.GRU(input_size, hidden_size, num_layers,
                               batch_first=True, bidirectional=bi_dir)
        else:
            self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                                batch_first=True, bidirectional=bi_dir)

    def forward(self, x):  # x: (batch,feature,seq)

        x = x.permute(0, 2, 1)

        batch_size = x.size(0)
        x, _ = self.lstm(x, self.__get_init_state(batch_size))  # x: (batch,seq,hidden)

        x = x.permute(0, 2, 1)

        return x

    def __get_init_state(self, batch_size):

        if self.bi_dir:
            nl_x_nd = 2 * self.num_layers
        else:
            nl_x_nd = 1 * self.num_layers

        h0 = torch.zeros(nl_x_nd, batch_size, self.hidden_size)
        h0 = h0.cuda()

        if self.use_gru:
            return h0
        else:
            c0 = torch.zeros(nl_x_nd, batch_size, self.hidden_size)
            c0 = c0.cuda()
            return (h0, c0)


class ChannelNorm(nn.Module):
    def __init__(self):
        super(ChannelNorm, self).__init__()

    def forward(self, x):  # (batch, feature, seq)
        divider = torch.max(torch.max(torch.abs(x), dim=0)[0], dim=1)[0] + 1e-5
        divider = divider.unsqueeze(0).unsqueeze(2)
        divider = divider.repeat(x.size(0), 1, x.size(2))
        x = x / divider
        return x


class Encoder(nn.Module):
    def __init__(self, v_or_k, input_size_v, input_size_k,
                 layer_type, layer_sizes,
                 kernel_size=None, norm_type=None,
                 downsample=True):
        super(Encoder, self).__init__()

        if layer_type not in ['TempConv', 'Bi-LSTM']:
            raise Exception('Invalid Layer Type')
        if layer_type == 'TempConv' and kernel_size is None:
            raise Exception('Kernel Size For TempConv Not Specified')

        self.output_size = layer_sizes[-1]

        module_list = []

        for layer in range(len(layer_sizes)):
            if layer == 0:
                if v_or_k == 0:
                    in_chl = input_size_v
                else:
                    in_chl = input_size_k
            else:
                in_chl = layer_sizes[layer - 1]
            out_chl = layer_sizes[layer]

            if layer_type == 'TempConv':
                conv_pad = kernel_size // 2
                module_list.append(('conv_{}'.format(layer),
                                    nn.Conv1d(in_chl, out_chl, kernel_size, padding=conv_pad)))
            elif layer_type == 'Bi-LSTM':
                module_list.append(('lstm_{}'.format(layer),
                                    LSTM_Layer(in_chl, out_chl // 2, 1, bi_dir=True)))

            if norm_type == 'Channel':
                module_list.append(('cn_{}'.format(layer),
                                    ChannelNorm()))
            elif norm_type == 'Batch':
                module_list.append(('bn_{}'.format(layer),
                                    nn.BatchNorm1d(out_chl)))
            elif norm_type == 'Instance':
                module_list.append(('in_{}'.format(layer),
                                    nn.InstanceNorm1d(out_chl)))
            else:
                print('No Norm Used!')

            if layer_type == 'TempConv':
                module_list.append(('relu_{}'.format(layer),
                                    nn.ReLU()))
            else:
                pass

            if downsample:
                module_list.append(('pool_{}'.format(layer),
                                    nn.MaxPool1d(kernel_size=2, stride=2)))

        self.module = nn.Sequential(OrderedDict(module_list))

    def forward(self, x):  # x: (batch,feature, seq)
        out = self.module(x)
        return out


class Decoder(nn.Module):
    def __init__(self, input_size,
                 layer_type, layer_sizes,
                 kernel_size=None, transposed_conv=None,
                 norm_type=None):
        super(Decoder, self).__init__()

        if layer_type not in ['TempConv', 'Bi-LSTM']:
            raise Exception('Invalid Layer Type')
        if layer_type == 'TempConv' and kernel_size is None:
            raise Exception('Kernel Size For TempConv Not Specified')
        if layer_type == 'TempConv' and transposed_conv is None:
            raise Exception('If Use Transposed Conv Not Specified')

        self.output_size = layer_sizes[-1]

        module_list = []

        for layer in range(len(layer_sizes)):
            if layer == 0:
                in_chl = input_size
            else:
                in_chl = layer_sizes[layer - 1]
            out_chl = layer_sizes[layer]

            module_list.append(('up_{}'.format(layer),
                                nn.Upsample(scale_factor=2)))

            if layer_type == 'TempConv':
                conv_pad = kernel_size // 2
                if transposed_conv:
                    module_list.append(('conv_{}'.format(layer),
                                        nn.ConvTranspose1d(in_chl, out_chl, kernel_size,
                                                           padding=conv_pad)))
                else:
                    module_list.append(('conv_{}'.format(layer),
                                        nn.Conv1d(in_chl, out_chl, kernel_size,
                                                  padding=conv_pad)))
            elif layer_type == 'Bi-LSTM':
                module_list.append(('lstm_{}'.format(layer),
                                    LSTM_Layer(in_chl, out_chl // 2, 1, bi_dir=True)))

            if norm_type == 'Channel':
                module_list.append(('cn_{}'.format(layer),
                                    ChannelNorm()))
            elif norm_type == 'Batch':
                module_list.append(('bn_{}'.format(layer),
                                    nn.BatchNorm1d(out_chl)))
            elif norm_type == 'Instance':
                module_list.append(('in_{}'.format(layer),
                                    nn.InstanceNorm1d(out_chl)))
            else:
                print('No Norm Used!')

            if layer_type == 'TempConv':
                module_list.append(('relu_{}'.format(layer),
                                    nn.ReLU()))
            else:
                pass

        self.module = nn.Sequential(OrderedDict(module_list))

    def forward(self, x):  # x: (batch,feature, seq)
        out = self.module(x)
        return out


class EncoderDecoderNet(nn.Module):
    def __init__(self, v_or_k, hidden_state,
                 encoder_params,
                 decoder_params,
                 mid_lstm_params=None):

        super(EncoderDecoderNet, self).__init__()

        self.encoder = Encoder(v_or_k, **encoder_params)

        # 中间层纯粹按照需求添加一下
        self.middle_lstm = None
        if mid_lstm_params is not None:
            self.middle_lstm = LSTM_Layer(mid_lstm_params['input_size'],
                                          mid_lstm_params['hidden_size'],
                                          mid_lstm_params['layer_num'],
                                          bi_dir=False)  # batch_first

        self.decoder = Decoder(**decoder_params)

        # self.fc1 = nn.Linear(self.decoder.output_size, hidden_state)

    def forward(self, x):

        x = x.permute(0, 2, 1)

        x = self.encoder(x)
        if self.middle_lstm is not None:
            x = self.middle_lstm(x)

        x = self.decoder(x)
        x = x.permute(0, 2, 1)

        # x = self.fc1(x)

        return x


class TcnGcnNet(nn.Module):
    def __init__(self, class_num, hidden_state,
                 encoder_params,
                 decoder_params,
                 mid_lstm_params):

        super(TcnGcnNet, self).__init__()

        self.hidden_state = hidden_state

        self.g_graph = DGLGraph().to(torch.device('cuda'))
        self.g_graph.add_nodes(3)
        edge_list = [(0, 1), (0, 2),
                     (1, 0), (1, 2),
                     (2, 0), (2, 1)]
        e_src, e_dst = tuple(zip(*edge_list))
        self.g_graph.add_edges(e_src, e_dst)
        self.edge_type = torch.from_numpy(
            np.array(
                [0, 0,
                 1, 2,
                 1, 2]
            ))
        self.edge_norm = None
        # self.edge_norm = torch.from_numpy(
        #     np.array(
        #         [0.5, 0.5,
        #          1.0, 1.0,
        #          1.0, 1.0]
        #     )).unsqueeze(1).long()
        # mid_lstm_params 默认是None
        self.tcn_vision = EncoderDecoderNet(0, hidden_state,
                                            encoder_params,
                                            decoder_params,
                                            mid_lstm_params)
        self.tcn_left = EncoderDecoderNet(1, hidden_state,
                                          encoder_params,
                                          decoder_params,
                                          mid_lstm_params)
        self.tcn_right = EncoderDecoderNet(1, hidden_state,
                                           encoder_params,
                                           decoder_params,
                                           mid_lstm_params)
        self.lstm_left = nn.LSTM(7, hidden_state, 1, batch_first=True)
        self.lstm_right = nn.LSTM(7, hidden_state, 1, batch_first=True)
        nn.init.xavier_normal_(self.lstm_left.all_weights[0][0])
        nn.init.xavier_normal_(self.lstm_left.all_weights[0][1])
        nn.init.xavier_normal_(self.lstm_right.all_weights[0][0])
        nn.init.xavier_normal_(self.lstm_right.all_weights[0][1])

        self.gnn = RGCN(g=self.g_graph, edge_type=self.edge_type,
                        edge_norm=self.edge_norm, num_nodes=3, i_dim=hidden_state,
                        h_dim=hidden_state, out_dim=hidden_state, num_rels=3,
                        num_bases=-1, num_hidden_layers=0, use_cuda=True,
                        use_self_loop=False)

        self.fc = nn.Linear(self.hidden_state * 3, class_num)
        nn.init.xavier_uniform_(self.fc.weight)

        self.cnn_feature = cnn_feature('linear')
        d_model = 512
        attn = 'prob'
        factor = 5
        n_heads = 8
        dropout = 0.05
        d_ff = 512
        activation = 'gelu'
        e_layers = 4
        distil = None
        embed = 'fixed'
        # 192 是最后一个维度，也就是dim
        self.enc_embedding = DataEmbedding(64, d_model, embed, dropout)
        Attn = ProbAttention if attn == 'prob' else FullAttention
        Attn = AutoCorrelation
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

            norm_layer=my_Layernorm(d_model)
        )

        ##2 part
        self.decomp = series_decomp(3)
        self.enc_embedding_2 = DataEmbedding(192, d_model, embed, dropout)
        Attn = ProbAttention if attn == 'prob' else FullAttention
        Attn = AutoCorrelation
        self.encoder_all_2 = att_Encoder(
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

            norm_layer=my_Layernorm(d_model)
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
        self.kine_fc = torch.nn.Linear(512, 128)
        self.new_fc = torch.nn.Linear(512, args.num_classes)

        # forward only change kinetic embedding

    # 只使用vision信息
    # def forward(self, x_vision, x_kinematics, return_emb=False):
    #     x_kine = self.kine_enc_embedding(x_kinematics)
    #     x_kine, attn_kine = self.kine_encoder(x_kine, attn_mask=None)
    #     x_kine = self.kine_fc(x_kine)
    #     x_vision = self.tcn_vision(x_vision)
    #     # 最终需要使用的结果是x_left,x_right,x_vision:[batch_size,video_len,64]
    #
    #     x_fu = x_vision
    #     x_fu = self.enc_embedding(x_fu)
    #     x_fu, attns = self.encoder_all(x_fu, attn_mask=None)
    #     out = self.new_fc(x_fu)
    #
    #     if return_emb:
    #         return out, 0
    #     else:
    #         return out

    # #forward only change kinetic embedding
    # def forward(self, x_vision, x_kinematics, return_emb=False):
    #     x_kine = self.kine_enc_embedding(x_kinematics)
    #     x_kine, attn_kine = self.kine_encoder(x_kine, attn_mask=None)
    #     x_kine = self.kine_fc(x_kine)
    #     x_vision = self.tcn_vision(x_vision)
    #     # 最终需要使用的结果是x_left,x_right,x_vision:[batch_size,video_len,64]
    #
    #     x_fu = torch.cat((x_kine, x_vision), dim=2)
    #     x_fu = self.enc_embedding(x_fu)
    #     x_fu, attns = self.encoder_all(x_fu, attn_mask=None)
    #     out = self.new_fc(x_fu)
    #
    #     if return_emb:
    #         return out, 0
    #     else:
    #         return out

        # forward only change final graph network
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

    def forward(self, x_vision, x_kinematics, return_emb=False):
        batch_size, seq_len, _, _, _ = x_vision.shape
        # !!!!!非继承的tensor切记要移动到cuda中去
        x_visions = torch.Tensor(batch_size, seq_len, 1000).cuda()
        for i in range(seq_len):
            # x_feature在token learner之后是[batch_size,self.s,512]
            x_vision_single = self.cnn_feature(x_vision[:, i, :, :, :])
            x_visions[:, i, :] = x_vision_single



        x_vision = self.tcn_vision(x_visions)

        # 最终需要使用的结果是x_left,x_right,x_vision:[batch_size,video_len,64]

        x_fu = x_vision
        seasonal_init, trend_init = self.decomp(x_fu)



        seasonal_init = self.enc_embedding(seasonal_init)
        seasonal_init, attns = self.encoder_all(seasonal_init, attn_mask=None)
        trend_init = self.enc_embedding(trend_init)
        trend_init, attns = self.encoder_all(trend_init, attn_mask=None)
        out = self.new_fc(trend_init+seasonal_init)

        if return_emb:
            return out, 0
        else:
            return out