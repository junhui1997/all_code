import torch
import torch.nn as nn

def weight_init(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)
        nn.init.constant_(m.bias, 0)
    # 也可以判断是否为conv2d，使用相应的初始化方式
    elif isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    # 是否为批归一化层
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)

# input:[b,c,h,w]
# out: [b,c], out
# x*weight，经过注意力层遮罩后的输出
class SpatialAttention(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.convo = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=(1, 1), stride=1),
            nn.BatchNorm2d(1),
            nn.ReLU()
        )
        #self.convo.apply(weight_init)

        self.sgap = nn.AvgPool2d(2)

    def forward(self, x):
        # mx:[b,1,h,w]
        # avg:[b,1,h,w]
        # combiner:[b,2,h,w]
        # fmap:[b,1,h,w],同时利用了最大值和平均值信息，将两通道数据1*1卷积为单通道数据
        # weight:[b,1,h,w]，就是scale
        mx = torch.max(x, 1)[0].unsqueeze(1)
        avg = torch.mean(x, 1).unsqueeze(1)
        combined = torch.cat([mx, avg], dim=1)
        fmap = self.convo(combined)
        #fmap = torch.clamp(fmap,min=1e-2)
        weight_map = torch.sigmoid(fmap)
        #print(torch.isnan(fmap).any(), torch.isnan(weight_map).any(), fmap.shape)
        out = (x * weight_map).mean(dim=(-2, -1))

        return out, x * weight_map


class token_learner(nn.Module):
    def __init__(self, S) -> None:
        super().__init__()
        self.S = S
        self.tokenizers = nn.ModuleList([SpatialAttention() for _ in range(S)])

    def forward(self, x):
        B, C, _, _ = x.shape
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        Z = torch.Tensor(B, self.S, C).to(device)
        for i in range(self.S):
            Ai, _ = self.tokenizers[i](x)  # [B, C]
            Z[:, i, :] = Ai
        return Z



