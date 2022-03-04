import torch
import torch.nn as nn
import numpy as np
from model.resnet_ori import resnet50

class FCN(nn.Module):
    def __init__(self, n_channels, n_classes, pretrained_model=True):
        super(FCN, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.context_path = resnet50(pretrained=False)
        if pretrained_model:
            self.context_path.load_state_dict(torch.load(pretrained_model),False)
        self.scores3 = nn.Conv2d(in_channels=2048, out_channels=n_classes, kernel_size=1)
        self.scores2 = nn.Conv2d(in_channels=1024, out_channels=n_classes, kernel_size=1)
        self.scores1 = nn.Conv2d(in_channels=512, out_channels=n_classes, kernel_size=1)
        #
        # # N=(w-1)xs+k-2p
        self.upsamplex8 = nn.ConvTranspose2d(in_channels=n_classes, out_channels=n_classes, kernel_size=16,
                                             stride=8, padding=4, bias=False)
        self.upsamplex8.weight.data = self.bilinear_kernel(n_classes, n_classes, 16)
        self.upsamplex16 = nn.ConvTranspose2d(in_channels=n_classes, out_channels=n_classes, kernel_size=4,
                                              stride=2, padding=1, bias=False)
        self.upsamplex16.weight.data = self.bilinear_kernel(n_classes, n_classes, 4)
        self.upsamplex32 = nn.ConvTranspose2d(in_channels=n_classes, out_channels=n_classes, kernel_size=2,
                                              stride=2, padding=0, bias=False)
        self.upsamplex32.weight.data = self.bilinear_kernel(n_classes, n_classes, 2)

    def forward(self, x):
        context_blocks = self.context_path(x)

        s3 = self.scores3(context_blocks[3])
        s3 = self.upsamplex32(s3)

        s2 = self.scores2(context_blocks[2])
        s2 = s2 + s3
        s2 = self.upsamplex16(s2)

        s1 = self.scores1(context_blocks[1])
        s = s1 + s2
        s = self.upsamplex8(s)

        return s

    def bilinear_kernel(self, in_channels, out_channels, kernel_size):
        '''
        return a bilinear filter tensor
        '''
        factor = (kernel_size + 1) // 2
        if kernel_size % 2 == 1:
            center = factor - 1
        else:
            center = factor - 0.5
        og = np.ogrid[:kernel_size, :kernel_size]
        filt = (1 - abs(og[0] - center) / factor) * (1 - abs(og[1] - center) / factor)
        weight = np.zeros((in_channels, out_channels, kernel_size, kernel_size), dtype='float32')
        weight[range(in_channels), range(out_channels), :, :] = filt
        return torch.from_numpy(weight)

if __name__ == "__main__":
    model = FCN(n_channels=3, n_classes=8)
    model.eval()
    image = torch.randn(1, 3, 256, 256)

    print(image.shape)
    print("input:", image.shape)
    print("output:", model(image).shape)