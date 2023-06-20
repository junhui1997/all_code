import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft
from layers.Embed import DataEmbedding
from layers.block import Block_1d, PermuteLayer, GRN_1d, ConvLayer_m





class fcn_mn(nn.Module):
    def __init__(self, configs):
        super(fcn_mn, self).__init__()
        self.e_layer = configs.e_layers
        self.d_model = configs.d_model
        self.seq_len = configs.seq_len
        self.down_sample = nn.ModuleList(
            [ConvLayer_m(self.d_model * (2 ** i), self.d_model * (2 ** (i + 1)), 'down') for i in
             range(configs.e_layers)])
        # 这里的d_model_f是上采样的终点
        self.d_model_f = self.d_model * (2 ** configs.e_layers)
        self.norm = nn.LayerNorm(self.d_model_f)
        self.grn = GRN_1d(self.d_model_f)
        self.up_sample = nn.ModuleList(
            [ConvLayer_m(self.d_model_f // (2 ** i), self.d_model_f // (2 ** (i + 1)), 'up') for i in
             range(configs.e_layers)])
        self.norm2 = nn.LayerNorm(self.d_model)
        self.grn2 = GRN_1d(self.d_model)
    def forward(self, x):
        inp = x
        for i in range(self.e_layer):
            x = self.down_sample[i](x)
        x = self.grn(x)
        x = self.norm(x)
        for i in range(self.e_layer):
            x = self.up_sample[i](x)
        # x = self.grn2(x)
        # x = self.norm2(x)
        padding_type = 'zero'
        padding_type = 'interpolate'
        if padding_type == 'zero':
            # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            # padding_part = torch.zeros([x.shape[0], self.seq_len-x.shape[1], x.shape[2]]).to(device)
            # x = torch.cat((x, padding_part), dim=1)
            x = x.permute(0, 2, 1)
            x = F.pad(x, (0, self.seq_len-x.shape[2]))
            x = x.permute(0, 2, 1)
        elif padding_type == 'interpolate':
            x = x.permute(0, 2, 1)
            x = F.interpolate(x, size=[self.seq_len], mode='linear')
            x = x.permute(0, 2, 1)
        x = x+inp
        return x


class Model(nn.Module):
    """
    ...
    """

    def __init__(self, configs):
        super(Model, self).__init__()
        self.configs = configs
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.label_len = configs.label_len
        self.pred_len = configs.pred_len
        # e_layer这里是2，注意这里是用for的写法，所以time_block不是共享的参数
        self.model = fcn_mn(configs)
        # embed：timeF， freq:'h'按小时进行的embedding, 这里的d_model没有按照公式上面进行计算，同时需要注意这个d_model特别小，不是512
        self.enc_embedding = DataEmbedding(configs.enc_in, configs.d_model, configs.embed, configs.freq,
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
        enc_out = self.enc_embedding_n(x_enc)  # [B,T,C]
        # enc_out = self.enc_embedding(x_enc, None)  # [B,T,C]
        enc_out = self.model(enc_out)

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
        return None

