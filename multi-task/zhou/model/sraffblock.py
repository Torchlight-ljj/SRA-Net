import torch
import torch.nn as nn
from torchvision.models.utils import load_state_dict_from_url
import numpy as np
import cv2
import torch.nn.functional as F
from math import sqrt

class SE_Block(nn.Module):
    def __init__(self, ch_in, reduction=4):
        super(SE_Block, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)				# 全局自适应池化
        
        
        self.fc = nn.Sequential(
            nn.Linear(ch_in, ch_in // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(ch_in // reduction, ch_in, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x) + x

class OursAB(nn.Module):
    def __init__(self, ch_in, reduction=8, c1=0.5):
        super(OursAB, self).__init__()
        self.c1 = c1
        self.avg_pool = nn.AdaptiveAvgPool2d(1)				# 全局自适应池化
        self.max_pool = nn.AdaptiveMaxPool2d(1) 
        self.fc1 = nn.Sequential(
            nn.Linear(ch_in, ch_in // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(ch_in // reduction, ch_in, bias=False),
            # nn.Sigmoid()
        )
        self.fc2 = nn.Sequential(
            nn.Linear(ch_in, ch_in // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(ch_in // reduction, ch_in, bias=False),
            # nn.Sigmoid()
        )
    def forward(self, x):
        b, c, _, _ = x.size()
        y1 = self.avg_pool(x).view(b, c)
        y1 = self.fc1(y1).view(b, c, 1, 1)
        y1 = self.c1*y1

        y2 = self.max_pool(x).view(b, c)
        y2 = self.fc2(y2).view(b, c, 1, 1)
        y2 = (1-self.c1)*y2

        y = F.sigmoid(y1+y2)
        return x * y.expand_as(x) + x

class CANet(nn.Module):
    def __init__(self, ch_in, reduction=4):
        super(CANet, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)				# 全局自适应池化
        self.max_pool = nn.AdaptiveMaxPool2d(1) 
        
        self.fc = nn.Sequential(
            nn.Linear(ch_in, ch_in // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(ch_in // reduction, ch_in, bias=False),
            # nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y1 = self.avg_pool(x).view(b, c)
        y1 = self.fc(y1).view(b, c, 1, 1)

        y2 = self.max_pool(x).view(b, c)
        y2 = self.fc(y2).view(b, c, 1, 1)
        y = F.sigmoid(y1+y2)
        return x * y.expand_as(x) + x
# class SRlayer_(nn.Module):
#     def __init__(self,channel):
#         super(SRlayer_,self).__init__()
#         self.channel = channel
#         self.amp_conv = nn.Conv2d(channel, channel, kernel_size=3, stride=1,padding=1)
#         self.amp_bn  = nn.BatchNorm2d(channel)
#         self.amp_relu = nn.ReLU()
#         self.phase_conv = nn.Conv2d(channel, channel, kernel_size=3, stride=1,padding=1)
#         self.Relu = nn.ReLU()

#     def forward(self,x):
#         rfft = torch.fft.rfftn(x)
#         amp = torch.abs(rfft) + torch.exp(torch.tensor(-10))
#         log_amp = torch.log(amp)
#         phase = torch.angle(rfft)
#         amp_filter = self.amp_conv(log_amp)
#         amp_sr = log_amp - amp_filter
#         SR = torch.fft.irfftn((amp_sr+1j*phase),x.size())
#         SR = self.amp_bn(SR)
#         # amp_sr = self.amp_relu(SR)
#         SR = self.Relu(SR)
#         return x + SR
class SRlayer_(nn.Module):
    def __init__(self,channel):
        super(SRlayer_,self).__init__()
        self.channel = channel
        self.batch = 1
        self.output_conv = nn.Conv2d(2,3, kernel_size=1)
        self.bn  = nn.BatchNorm2d(3)
        self.Relu = nn.ReLU()
        
        nn.init.kaiming_normal_(self.output_conv.weight, mode='fan_out', nonlinearity='relu')
        
        self.kernalsize = 3
        self.amp_conv = nn.Conv2d(self.channel, self.channel, kernel_size=self.kernalsize, stride=1,padding=1, bias=False)
        self.fucker = np.zeros([self.kernalsize,self.kernalsize])
        for i in range(self.kernalsize):
            for j in range(self.kernalsize):
                self.fucker[i][j] = 1/np.square(self.kernalsize)
        self.aveKernal = torch.Tensor(self.fucker).unsqueeze(0).unsqueeze(0).repeat(self.batch,self.channel,1,1)
        self.amp_conv.weight = nn.Parameter(self.aveKernal, requires_grad=False)
        
        self.amp_relu = nn.ReLU()
        self.gaussi = nn.Conv2d(self.channel, self.channel, kernel_size=3, stride=1,padding=1, bias=False)
        self.gauKernal = torch.Tensor([[1/16, 1/8, 1/16], [1/8, 1/4, 1/8], [1/16, 1/8, 1/16]]).unsqueeze(0).unsqueeze(0).repeat(self.batch,self.channel,1,1)
        self.gaussi.weight = nn.Parameter(self.gauKernal,requires_grad=False)

    def forward(self,x):
        out = []
        for batch in range(x.shape[0]):
            x1 = x[batch,0,:,:].unsqueeze(0).unsqueeze(0)
            rfft = torch.fft.fftn(x1)
            amp = torch.abs(rfft) + torch.exp(torch.tensor(-10))
            log_amp = torch.log(amp)
            phase = torch.angle(rfft)
            amp_filter = self.amp_conv(log_amp)
            amp_sr = log_amp - amp_filter
            SR = torch.fft.ifftn(torch.exp(amp_sr+1j*phase))
            SR = torch.abs(SR)
            SR = self.gaussi(SR)
            y = torch.cat([SR,x1],dim=1)
            y = self.output_conv(y)
            y = self.bn(y)
            y = self.Relu(y)

            out.append(y)
        
        return torch.cat(out,dim=0)
        # return x.repeat(1,3,1,1)
class ASPP(nn.Module):
    def __init__(self, in_channel=1, depth=100, k_size = 3):
        super(ASPP,self).__init__()
        #F = 2*（dilation -1 ）*(kernal -1) + kernal

        self.atrous_block1 = nn.Conv2d(in_channel, depth, 1, 1)
        self.atrous_block6 = nn.Conv2d(in_channel, depth, k_size, 1, padding=2, dilation=2)
        self.atrous_block12 = nn.Conv2d(in_channel, depth, k_size, 1, padding=4, dilation=4)
        self.atrous_block18 = nn.Conv2d(in_channel, depth, k_size, 1, padding=8, dilation=8)
        self.atrous_block24 = nn.Conv2d(in_channel, depth, k_size, 1, padding=16, dilation=16)
        # self.atrous_block30 = nn.Conv2d(in_channel, depth, k_size, 1, padding=10, dilation=10)
        # self.atrous_block36 = nn.Conv2d(in_channel, depth, k_size, 1, padding=12, dilation=12)
        # self.atrous_block42 = nn.Conv2d(in_channel, depth, k_size, 1, padding=14, dilation=14)
        self.conv_1x1_output = nn.Conv2d(depth * 5, depth, 1, 1)
 
    def forward(self, x):
        atrous_block1 = self.atrous_block1(x)
        atrous_block6 = self.atrous_block6(x)
        atrous_block12 = self.atrous_block12(x)
        atrous_block18 = self.atrous_block18(x)
        atrous_block24 = self.atrous_block24(x)
        # atrous_block30 = self.atrous_block30(x)
        # atrous_block36 = self.atrous_block36(x)
        # atrous_block42 = self.atrous_block42(x)
 
        net = self.conv_1x1_output(torch.cat([atrous_block1, atrous_block6,
                                              atrous_block12, atrous_block18,atrous_block24], dim=1))
        return net    

class FullAttention(nn.Module):
    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=True):
        super(FullAttention, self).__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)
        
    def forward(self, queries, keys, values):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        scale = self.scale or 1./sqrt(E)

        scores = torch.einsum("blhe,bshe->bhls", queries, keys)

        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        V = torch.einsum("bhls,bshd->blhd", A, values)
        
        if self.output_attention:
            return (V.contiguous(), A)
        else:
            return (V.contiguous(), None)
        
class AttentionLayer(nn.Module):
    def __init__(self, attention, d_model, n_heads, 
                 d_keys=None, d_values=None, mix=False):
        super(AttentionLayer, self).__init__()

        d_keys = d_keys or (d_model//n_heads)
        d_values = d_values or (d_model//n_heads)

        self.inner_attention = attention
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads
        self.mix = mix

    def forward(self, queries, keys, values):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads

        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)

        out, attn = self.inner_attention(
            queries,
            keys,
            values
        )
        if self.mix:
            out = out.transpose(2,1).contiguous()
        out = out.view(B, L, -1)

        return self.out_projection(out), attn


