""" Parts of the U-Net model """

import torch
import torch.nn as nn
import torch.nn.functional as F
# from resnet import resnet50
from resnet import resnet50
import torch.nn.functional as F
from math import sqrt
import numpy  as np
import time
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
    def __init__(self, attention, d_model, out_channels, n_heads, 
                 d_keys=None, d_values=None, mix=False):
        super(AttentionLayer, self).__init__()

        d_keys = d_keys or (d_model//n_heads)
        d_values = d_values or (d_model//n_heads)

        self.inner_attention = attention
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, out_channels)
        self.n_heads = n_heads
        self.mix = mix

    def forward(self, queries, keys, values):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads
        # print(queries.shape)
        # print(keys.shape)
        # print(values.shape)
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

class OursAB(nn.Module):
    def __init__(self, ch_in, reduction=8, c1=0.5):
        super(OursAB, self).__init__()
        self.c1 = nn.Parameter(torch.Tensor([c1]))
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
class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up_attention(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
        self.out_channels = out_channels
        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_channels // 2, in_channels // 2, kernel_size=2, stride=2)

        # self.conv = DoubleConv(in_channels, out_channels)
        self.attention = AttentionLayer(FullAttention(), d_model = in_channels, out_channels = self.out_channels, n_heads = 8,)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        B,C,W,H = x2.shape
        # input is CHW
        diffY = torch.tensor([x2.size()[2] - x1.size()[2]])
        diffX = torch.tensor([x2.size()[3] - x1.size()[3]])

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x1 = torch.flatten(x1,2).transpose(1,2)
        x2 = torch.flatten(x2,2).transpose(1,2)
        out,_ = self.attention(x2,x1,x1)
        out = out.transpose(1,2)
        out = torch.reshape(out,(B,self.out_channels,W,H))
        return out
        # while True:
        #     pass
        # x = torch.cat([x2, x1], dim=1)
        # return self.conv(x)
class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_channels // 2, in_channels // 2, kernel_size=2, stride=2)

        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = torch.tensor([x2.size()[2] - x1.size()[2]])
        diffX = torch.tensor([x2.size()[3] - x1.size()[3]])

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x1 = self.conv(x)
        return x1

class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 512)
        self.up1 = Up(1024, 256, bilinear)
        self.up2 = Up(512, 128, bilinear)
        self.up3 = Up(256, 64, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits

class UNet_Res(nn.Module):
    def __init__(self, n_channels, n_classes, pretrained_model_path = None, bilinear=True):
        super(UNet_Res, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        # self.weights_new = self.state_dict()
        self.context_path = resnet50(pretrained=False,num_classes=4)
        
        # self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        # self.fc = nn.Linear(2048, n_classes)
        if pretrained_model_path:
            # self.load_weights(pretrained_model_path)
            self.context_path.load_state_dict(torch.load(pretrained_model_path),False)
        self.refine = nn.Conv2d(2048, 1024, kernel_size=3, padding=1)
        self.up1 = Up(2048, 512, bilinear)
        # self.attention = AttentionLayer(FullAttention(), d_model = 512, out_channels = 512, n_heads = 8,)
        self.up2 = Up(1024, 256, bilinear)
        self.up3 = Up(512, 128, bilinear)
        self.outc = OutConv(128, n_classes)
        
        self.SR= SRlayer_(1)
        # self.w1 = nn.Parameter(torch.Tensor([0.5])).cuda()
        # self.w2 = nn.Parameter(torch.Tensor([0.5])).cuda()
        # self.w3 = nn.Parameter(torch.Tensor([0.5])).cuda()

    def forward(self, x):
        x = self.SR(x)
        context_blocks,y_cla,features = self.context_path(x)
        context_blocks.reverse()
        # print(context_blocks[0].shape,context_blocks[1].shape,context_blocks[2].shape,context_blocks[3].shape)
        # while True:
        #     pass
        # y_cla = self.avgpool(context_blocks[0]).flatten(1)
        # y_cla = self.fc(y_cla) 

        y = self.refine(context_blocks[0])
        y = self.up1(y, context_blocks[1])
        # size = y.shape
        # y = torch.flatten(y,2).transpose(1,2)
        # y,_ = self.attention(y,y,y)
        # y = y.transpose(1,2)
        # y = torch.reshape(y,size)
        y = self.up2(y, context_blocks[2])
        y = self.up3(y, context_blocks[3])
        y = self.outc(y)
        y = F.interpolate(y, scale_factor=4,
                          mode='bilinear',
                          align_corners=True)
        return y,y_cla,features

    def load_weights(self, weights_path):
        weights = torch.load(weights_path)
        del weights["fc.weight"]
        del weights["fc.bias"]

        names = []
        for key, value in self.context_path.state_dict().items():
            if "num_batches_tracked" in key:
                continue
            names.append(key)

        for name, dict in zip(names, weights.items()):
            self.weights_new[name] = dict[1]

        self.context_path.load_state_dict(self.weights_new,False)

if __name__ == "__main__":
    model = UNet_Res(n_channels=3, n_classes=4,pretrained_model_path=None).cuda()
    model.eval()
    image = torch.randn(1, 3, 512, 512).cuda()
    start = time.time()
    seg,cla,fea = model(image)
    end = time.time()
    print(1/(end-start))
    # model(image).shape
    # print(image.shape)
    # print("input:", image.shape)
    # print("output:", model(image)[0].shape,model(image)[1].shape)