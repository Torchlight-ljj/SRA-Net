""" Parts of the U-Net model """

import torch
import torch.nn as nn
import torch.nn.functional as F
# from resnet import resnet50
from model.resnet_ori import resnet50
import torch.nn.functional as F
from math import sqrt
import numpy  as np
import time
from model.vgg import vgg16
# import parameters
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

    def forward(self, x):
        context_blocks = self.context_path(x)
        context_blocks.reverse()
        y = self.refine(context_blocks[0])
        y = self.up1(y, context_blocks[1])
        y = self.up2(y, context_blocks[2])
        y = self.up3(y, context_blocks[3])
        y = self.outc(y)
        y = F.interpolate(y, scale_factor=4,
                          mode='bilinear',
                          align_corners=True)
        return y

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
#     image = torch.randn(1, 3, 512, 512)
#     model = UNet_Res(n_channels=3, n_classes=4, pretrained_model_path=None)
#     vgg_model = vgg16(pretrained=False)
#     print(parameters.count_params(model,input_size=512))
#     print(parameters.count_params(vgg_model,input_channel=4,input_size=512))
    # vgg_model.load_state_dict(torch.load('../vgg16.pth'),False)
    # model.eval()
    # vgg_image = torch.randn(1,4,512,512)
    # vgg_model.eval()
    # out = vgg_model(vgg_image)
    # print(model(image).shape,out[0].shape,out[1].shape,out[2].shape,out[3].shape)

    # start = time.time()
    # seg,cla,fea = model(image)
    # end = time.time()
    # print(1/(end-start))
    # model(image).shape
    # print(image.shape)
    # print("input:", image.shape)
    # print("output:", model(image)[0].shape,model(image)[1].shape)features.0.weight