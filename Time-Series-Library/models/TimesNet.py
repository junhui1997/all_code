import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft
from layers.Embed import DataEmbedding
from layers.Conv_Blocks import Inception_Block_V1, Inception_Block_V2, Inception_Block_V3

"""
    对输入的时间序列 `x` 进行实数值 FFT 变换，并根据幅值选取前 k 个频率，计算时间序列的周期和幅值。

    Args:
        x (torch.Tensor): 输入的时间序列张量，形状为 `[B, T, C]`，其中 `B` 表示批量大小，`T` 表示时间序列长度，`C` 表示通道数。
        k (int, optional): 选取幅值最大的前 k 个频率。默认值为 2。

    Returns:
        一个元组，包含两个张量：`period` 和 `amplitudes`。
        - period (ndarray): 形状为 `[k]` 的张量，表示时间序列的周期。每个元素是整数，是将时间序列长度 `T` 整除 `k` 中对应的值的整数部分。
        - amplitudes (torch.Tensor): 形状为 `[B, k]` 的张量，表示时间序列在前 k 个幅值最大的频率上的幅值。

    Example:
        >>> x = torch.randn(32, 256, 64)
        >>> period, amplitudes = FFT_for_Period(x, k=3)
    """
def FFT_for_Period(x, k=2):
    # [B, T, C]
    # 这里xf：[batch_size,T/2+1，d_model]
    xf = torch.fft.rfft(x, dim=1)
    # find period by amplitudes
    frequency_list = abs(xf).mean(0).mean(-1)
    # 这里是为了避免自身的干扰，而且求得是绝对的amplitude，负值越大也意味着越大的变化
    frequency_list[0] = 0
    _, top_list = torch.topk(frequency_list, k)
    # 在下一次计算时候斩断计算图
    top_list = top_list.detach().cpu().numpy()
    # 根据几个频率最高峰来确认间隔，上面的top_list就是index，这个也侧面解释了为啥是他画的freq的那个图中对应频率只有前半段
    # 这里是整除
    period = x.shape[1] // top_list
    return period, abs(xf).mean(-1)[:, top_list]


class TimesBlock(nn.Module):
    def __init__(self, configs):
        super(TimesBlock, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.k = configs.top_k
        # parameter-efficient design
        # d_ff是卷积层的中间变量，因为使用了一个d_model->d_ff->d_model的变化
        # num_kernels默认是6
        self.conv = nn.Sequential(
            Inception_Block_V3(configs.d_model, configs.d_ff,
                               num_kernels=configs.num_kernels),
            nn.GELU(),
            Inception_Block_V3(configs.d_ff, configs.d_model,
                               num_kernels=configs.num_kernels)
        )

    def forward(self, x):
        # 这里的T对于forecast来说是seq_len+pred,对于其他任务来说就只是seq
        B, T, N = x.size()
        #period_list [k],period_wieght:[batch_size,k]
        period_list, period_weight = FFT_for_Period(x, self.k)

        res = []
        for i in range(self.k):
            period = period_list[i]
            # padding
            # out这里就是补出来0之后的
            if (self.seq_len + self.pred_len) % period != 0:
                # length是周期整除之后的数值，相当于是整除后加一，给额外补了一截0为了凑周期
                length = (((self.seq_len + self.pred_len) // period) + 1) * period
                padding = torch.zeros([x.shape[0], (length - (self.seq_len + self.pred_len)), x.shape[2]]).to(x.device)
                out = torch.cat([x, padding], dim=1)
            else:
                # 刚好被周期整除时候
                length = (self.seq_len + self.pred_len)
                out = x
            # reshape
            # length//period得出的是一共被切成几段，这个维度上面代表的是inter-period
            # 在period这个维度上是intra_period
            # out:[batch_size, d_model,length//period,period]
            out = out.reshape(B, length // period, period, N).permute(0, 3, 1, 2).contiguous()
            # 2D conv: from 1d Variation to 2d Variation
            out = self.conv(out)
            # reshape back，还原回去了[batch_size,padded_len,d_model]
            out = out.permute(0, 2, 3, 1).reshape(B, -1, N)
            # 把padding 部分给去掉了
            res.append(out[:, :(self.seq_len + self.pred_len), :])
        # res [batch_size,seq_len+pred_len,d_model,k]
        res = torch.stack(res, dim=-1)
        # adaptive aggregation
        # periond_weight[batch_size,k],上面生成的本来就是这个维度
        period_weight = F.softmax(period_weight, dim=1)
        # period_weight:[batch_size,seq_len+pred,d_model,k]
        period_weight = period_weight.unsqueeze(1).unsqueeze(1).repeat(1, T, N, 1)
        # res*period_weight:[batch_size,seq_len+pred_len,d_model,k]*[batch_size,seq_len+pred,d_model,k], 这里是element-wise的
        # 之后消去了最后一个维度k，其实间接相当于取均值了，相当于是不加权平均
        res = torch.sum(res * period_weight, -1)
        # residual connection
        res = res + x
        return res


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
        self.model = nn.ModuleList([TimesBlock(configs)
                                    for _ in range(configs.e_layers)])
        # embed：timeF， freq:'h'按小时进行的embedding, 这里的d_model没有按照公式上面进行计算，同时需要注意这个d_model特别小，不是512
        self.enc_embedding = DataEmbedding(configs.enc_in, configs.d_model, configs.embed, configs.freq,
                                           configs.dropout)
        self.layer = configs.e_layers
        self.layer_norm = nn.LayerNorm(configs.d_model)
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            # 只有预测任务有predict_linear，这里相当于是不使用encoder-decoder structure就能实现生成式模型
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
        # Normalization from Non-stationary Transformer
        # x_enc:[batch_size,seq_len,dim]
        # 这里对seq去了均值，之后每个seq都减去了均值
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        # 这里使用的是有偏估计，只关心估值方向不关心具体数值的话，使用有偏估计比无偏估计的更方便和高效
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev

        # embedding
        enc_out = self.enc_embedding(x_enc, x_mark_enc)  # [B,T,C]
        # seq_len -> seq_len+prediction
        enc_out = self.predict_linear(enc_out.permute(0, 2, 1)).permute(
            0, 2, 1)  # align temporal dimension
        # TimesNet
        for i in range(self.layer):
            enc_out = self.layer_norm(self.model[i](enc_out))
        # porject back
        dec_out = self.projection(enc_out)

        # De-Normalization from Non-stationary Transformer
        dec_out = dec_out * \
                  (stdev[:, 0, :].unsqueeze(1).repeat(
                      1, self.pred_len + self.seq_len, 1))
        dec_out = dec_out + \
                  (means[:, 0, :].unsqueeze(1).repeat(
                      1, self.pred_len + self.seq_len, 1))
        return dec_out

    def imputation(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask):
        # Normalization from Non-stationary Transformer
        means = torch.sum(x_enc, dim=1) / torch.sum(mask == 1, dim=1)
        means = means.unsqueeze(1).detach()
        x_enc = x_enc - means
        x_enc = x_enc.masked_fill(mask == 0, 0)
        stdev = torch.sqrt(torch.sum(x_enc * x_enc, dim=1) /
                           torch.sum(mask == 1, dim=1) + 1e-5)
        stdev = stdev.unsqueeze(1).detach()
        x_enc /= stdev

        # embedding
        enc_out = self.enc_embedding(x_enc, x_mark_enc)  # [B,T,C]
        # TimesNet
        for i in range(self.layer):
            enc_out = self.layer_norm(self.model[i](enc_out))
        # porject back
        dec_out = self.projection(enc_out)

        # De-Normalization from Non-stationary Transformer
        dec_out = dec_out * \
                  (stdev[:, 0, :].unsqueeze(1).repeat(
                      1, self.pred_len + self.seq_len, 1))
        dec_out = dec_out + \
                  (means[:, 0, :].unsqueeze(1).repeat(
                      1, self.pred_len + self.seq_len, 1))
        return dec_out

    def anomaly_detection(self, x_enc):
        # Normalization from Non-stationary Transformer
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(
            torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev

        # embedding
        enc_out = self.enc_embedding(x_enc, None)  # [B,T,C]
        # TimesNet
        for i in range(self.layer):
            enc_out = self.layer_norm(self.model[i](enc_out))
        # porject back
        dec_out = self.projection(enc_out)

        # De-Normalization from Non-stationary Transformer
        dec_out = dec_out * \
                  (stdev[:, 0, :].unsqueeze(1).repeat(
                      1, self.pred_len + self.seq_len, 1))
        dec_out = dec_out + \
                  (means[:, 0, :].unsqueeze(1).repeat(
                      1, self.pred_len + self.seq_len, 1))
        return dec_out

    def classification(self, x_enc, x_mark_enc):
        # embedding
        enc_out = self.enc_embedding(x_enc, None)  # [B,T,C]
        # TimesNet
        for i in range(self.layer):
            enc_out = self.layer_norm(self.model[i](enc_out))

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
            #从后往前数的那部分才是输入，原本完整的dec_out:[batch_size,label+pred,dim]
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
