

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft
from layers.Embed import DataEmbedding
from layers.block import lstm_n, fusion_layer
from layers.Autoformer_EncDec import series_decomp
from models.fcn_m import fcn_mn








class ConvLayer(nn.Module):
    def __init__(self, in_c, out_c, conv='down'):
        super(ConvLayer, self).__init__()
        # 这里应该是由于api版本的不同，想要维持卷积完后的形状不同需要进行的操作
        padding = 1 if torch.__version__ >= '1.5.0' else 2
        # 保持channel的维度不变,此时的cin是d_model
        if conv == 'down':
            self.Conv = nn.Conv1d(in_channels=in_c,
                                  out_channels=out_c,
                                  kernel_size=3,
                                  stride=2,
                                  padding=padding,
                                  padding_mode='circular')
        else:
            self.Conv = nn.ConvTranspose1d(in_channels=in_c,
                                           out_channels=out_c,
                                           kernel_size=3,
                                           stride=2,
                                           padding=padding,
                                           padding_mode='zeros')
        self.norm = nn.BatchNorm1d(out_c)
        # 和relu有细微差别
        self.activation = nn.ReLU()

    def forward(self, x):
        # 这一步同样是为了对d_model进行卷积
        x = self.Conv(x.permute(0, 2, 1))
        x = self.norm(x)
        x = self.activation(x)
        # 交换回来第一步卷积交换过去的数值
        x = x.transpose(1, 2)
        return x


class fcn_n(nn.Module):
    def __init__(self, configs):
        super(fcn_n, self).__init__()
        self.e_layer = configs.e_layers
        self.d_model = configs.d_model
        self.seq_len = configs.seq_len
        self.down_sample = nn.ModuleList(
            [ConvLayer(self.d_model * (2 ** i), self.d_model * (2 ** (i + 1)), 'down') for i in
             range(configs.e_layers)])
        # 这里的d_model_f是上采样的终点
        self.d_model_f = self.d_model * (2 ** configs.e_layers)
        self.up_sample = nn.ModuleList(
            [ConvLayer(self.d_model_f // (2 ** i), self.d_model_f // (2 ** (i + 1)), 'up') for i in
             range(configs.e_layers)])

    def forward(self, x):
        for i in range(self.e_layer):
            x = self.down_sample[i](x)
        for i in range(self.e_layer):
            x = self.up_sample[i](x)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        padding_part = torch.zeros([x.shape[0], self.seq_len-x.shape[1], x.shape[2]]).to(device)
        x = torch.cat((x, padding_part), dim=1)
        return x



class lstm_fcn_n(nn.Module):
    def __init__(self, configs):
        super(lstm_fcn_n, self).__init__()
        kernel_size = configs.moving_avg
        self.decomp = series_decomp(kernel_size)
        self.fcn = fcn_mn(configs)
        self.lstm = lstm_n(configs)
        self.fusion = fusion_layer(configs, 'weight_sum', '')
        self.norm = nn.LayerNorm(configs.d_model)

    def forward(self, x):
        # 两个lstm效果并不好，使用lion后好一些
        seasonal_init, trend_init = self.decomp(x)
        x_fcn = self.fcn(seasonal_init)
        x_lstm = self.lstm(trend_init)
        # x_lstm = self.norm(x_lstm)
        # x_fcn = self.fcn(x)
        # x_lstm = self.lstm(x)
        out = self.fusion(x_fcn, x_lstm)
        return out

class Model(nn.Module):
    """
    Paper link: https://openreview.net/pdf?id=ju_Uqw384Oq
    """

    def __init__(self, configs):
        super(Model, self).__init__()
        self.configs = configs
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.label_len = configs.label_len
        self.pred_len = configs.pred_len
        # e_layer这里是2，注意这里是用for的写法，所以time_block不是共享的参数
        self.model1 = lstm_fcn_n(configs)
        self.model2 = lstm_fcn_n(configs)
        self.enc = configs.enc_in//2
        self.fusion = fusion_layer(configs, 'former', 'prob')
        # embed：timeF， freq:'h'按小时进行的embedding, 这里的d_model没有按照公式上面进行计算，同时需要注意这个d_model特别小，不是512
        self.enc_embedding_f = DataEmbedding(configs.enc_in//2, configs.d_model, configs.embed, configs.freq,
                                           configs.dropout)
        self.enc_embedding_p = DataEmbedding(configs.enc_in // 2, configs.d_model, configs.embed, configs.freq,
                                           configs.dropout)
        self.enc_embedding_n = nn.Linear(configs.enc_in, configs.d_model)
        self.layer = configs.e_layers
        self.layer_norm = nn.LayerNorm(configs.d_model)
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            # 只有预测任务有predict_linear
            self.predict_linear = nn.Linear(self.seq_len, self.pred_len + self.seq_len)
            # bias这个bool量决定是否引入偏置项
            # projection是最后的输出层了
            self.projection = nn.Linear(
                configs.d_model, configs.c_out, bias=True)
        if self.task_name == 'imputation' or self.task_name == 'anomaly_detection':
            self.projection = nn.Linear(
                configs.d_model, configs.c_out, bias=True)
        if self.task_name == 'classification':
            self.act = F.gelu
            self.dropout = nn.Dropout(configs.dropout)
            self.projection = nn.Linear(
                configs.d_model * configs.seq_len, configs.num_class)

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):

        # embedding
        enc_out = self.enc_embedding(x_enc, x_mark_enc)  # [B,T,C]
        # seq_len -> seq_len+prediction
        enc_out = self.predict_linear(enc_out.permute(0, 2, 1)).permute(
            0, 2, 1)  # align temporal dimension
        # TimesNet
        enc_out = self.model(enc_out)
        # porject back
        dec_out = self.projection(enc_out)

        return dec_out

    def imputation(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask):
        print(' do not support this task')
        return 0

    def anomaly_detection(self, x_enc):
        print(' do not support this task')
        return 0

    def classification(self, x_enc, x_mark_enc):
        # embedding
        enc_f = x_enc[:, :, 0:self.enc]
        enc_p = x_enc[:, :, self.enc::]

        enc_out_f = self.enc_embedding_p(enc_f, None)  # [B,T,C]
        enc_out_f = self.model1(enc_out_f)
        enc_out_p = self.enc_embedding_p(enc_p, None)  # [B,T,C]
        enc_out_p = self.model2(enc_out_p)
        enc_out = self.fusion(enc_out_f,enc_out_p)
        # Output
        # the output transformer encoder/decoder embeddings don't include non-linearity
        output = self.act(enc_out)
        output = self.dropout(output)
        # zero-out padding embeddings
        output = output * x_mark_enc.unsqueeze(-1)
        # (batch_size, seq_length * d_model)
        output = output.reshape(output.shape[0], -1)
        output = self.projection(output)  # (batch_size, num_classes)
        return output

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
            # 从后往前数的那部分才是输入，原本完整的dec_out:[batch_size,label+pred,dim]
            return dec_out[:, -self.pred_len:, :]  # [B, L, D]
        if self.task_name == 'imputation':
            dec_out = self.imputation(
                x_enc, x_mark_enc, x_dec, x_mark_dec, mask)
            return dec_out  # [B, L, D]
        if self.task_name == 'anomaly_detection':
            dec_out = self.anomaly_detection(x_enc)
            return dec_out  # [B, L, D]
        if self.task_name == 'classification':
            dec_out = self.classification(x_enc, x_mark_enc)
            return dec_out  # [B, N]
        if self.task_name[:6] == 'encoder':
            dec_out = self.encoding(x_enc, x_mark_enc)
            return dec_out  # [B, L, D]
        return None
