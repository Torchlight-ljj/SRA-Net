""" Parts of the U-Net model """

import torch
import torch.nn as nn
import torch.nn.functional as F
# from resnet import resnet50
import torch.nn.functional as F
# import parameters
import time
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

class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, cla_n_classes=4, bilinear=True):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 512)
        self.avePool = nn.AdaptiveAvgPool2d((1,1)) 
        self.fc1 = nn.Linear(1280,2048)
        self.fc2 = nn.Linear(2048,cla_n_classes)
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
        a1 = self.avePool(x4)#512
        x5 = self.down4(x4)
        a2 = self.avePool(x5)#512
        x = self.up1(x5, x4)#256
        a3 = self.avePool(x)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        a_cat = torch.cat([a1,a2,a3],dim=1).flatten(1)
        a = F.relu(self.fc1(a_cat))
        a = F.relu(self.fc2(a))
        logits = self.outc(x)
        return logits,a


# if __name__ == "__main__":
#     model = UNet(n_channels=3, n_classes=4)
#     model.eval()
#     image = torch.randn([1, 3, 512, 512])
#     # image.cuda()
#     start = time.time()
#     seg,cla = model(image)
#     end = time.time()
#     print(round(1/(end-start),3))
#     paras,flops = parameters.count_params(model,input_size=512)
#     nParams = sum([p.nelement() for p in model.parameters()])
#     print('*unet: number of parameters: %.3fM FLOPS: %.3fM. fps:%.3f.' % (round(paras,3),round(flops,3),round(1/(end-start),3)))
    # model(image).shape
    # print(image.shape)
    # print("input:", image.shape)
    # print("output:", model(image)[0].shape,model(image)[0].shape)