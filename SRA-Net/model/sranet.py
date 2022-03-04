import torch
import torch.nn as nn
import torch.nn.functional as F
from model.resnet import resnet50
import torch.nn.functional as F
from math import sqrt
import numpy  as np
import time
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

class UNet_Res(nn.Module):
    def __init__(self, n_channels, n_classes, pretrained_model_path = None, bilinear=True):
        super(UNet_Res, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.context_path = resnet50(pretrained=False,num_classes=4)
        
        if pretrained_model_path:
            self.context_path.load_state_dict(torch.load(pretrained_model_path),False)
        self.refine = nn.Conv2d(2048, 1024, kernel_size=3, padding=1)
        self.up1 = Up(2048, 512, bilinear)
        self.up2 = Up(1024, 256, bilinear)
        self.up3 = Up(512, 128, bilinear)
        self.outc = OutConv(128, n_classes)
        self.SR= SRlayer_(1)


    def forward(self, x):
        x = self.SR(x)
        context_blocks,y_cla,features = self.context_path(x)
        context_blocks.reverse()
        y = self.refine(context_blocks[0])
        y = self.up1(y, context_blocks[1])
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

# if __name__ == "__main__":
#     model = UNet_Res(n_channels=3, n_classes=2,pretrained_model_path=None).cuda()
#     # model.eval()
#     image = torch.randn(1, 3, 512, 512).cuda()
#     start = time.time()
#     seg,cla,fea = model(image)
#     end = time.time()
#     print(1/(end-start))
#     # paras,flops = parameters.count_params(model,input_size=512)
#     # print('*unet: number of parameters: %.3fM FLOPS: %.3fM. fps:%.3f.' % (round(paras,3),round(flops,3),round(1/(end-start),3)))    
    
    
#     nParams = sum([p.nelement() for p in model.parameters()])
#     print(nParams)