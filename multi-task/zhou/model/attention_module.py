import torch
import torch.nn as nn
from torch.nn import Module, Sequential, Conv2d, ReLU,AdaptiveMaxPool2d, AdaptiveAvgPool2d, \
    NLLLoss, BCELoss, CrossEntropyLoss, AvgPool2d, MaxPool2d, Parameter, Linear, Sigmoid, Softmax, Dropout, Embedding
from torch.nn import functional as F
from torch.autograd import Variable
#torch_ver = torch.__version__[:3]

class PAM_Module(Module):
    """ Position attention module"""

    # Ref from SAGAN
    def __init__(self, in_dim):
        super(PAM_Module, self).__init__()
        self.query_conv = Conv2d(in_channels=in_dim,
                                 out_channels=in_dim // 8,
                                 kernel_size=1)
        self.key_conv = Conv2d(in_channels=in_dim,
                               out_channels=in_dim // 8,
                               kernel_size=1)
        self.value_conv = Conv2d(in_channels=in_dim,
                                 out_channels=in_dim,
                                 kernel_size=1)
        self.gamma = Parameter(torch.zeros(1))

        self.softmax = Softmax(dim=-1)

    def forward(self, x):
        """
        inputs :
            x : input feature maps( N X C X H X W)
        returns :
            out : attention value + input feature
            attention: N X (HxW) X (HxW)
        """
        m_batchsize, C, height, width = x.size()

        # B => N, C, HW
        proj_query = self.query_conv(x).view(m_batchsize, -1, width * height)
        # B' => N, HW, C
        proj_query = proj_query.permute(0, 2, 1)

        # C => N, C, HW
        proj_key = self.key_conv(x).view(m_batchsize, -1, width * height)

        # B'xC => N, HW, HW
        energy = torch.bmm(proj_query, proj_key)
        # S = softmax(B'xC) => N, HW, HW
        attention = self.softmax(energy)

        # D => N, C, HW
        proj_value = self.value_conv(x).view(m_batchsize, -1, width * height)

        # DxS' => N, C, HW
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        # N, C, H, W
        out = out.view(m_batchsize, C, height, width)

        out = self.gamma * out + x
        return out

        
class PAM_SE_Module(Module):
    """ Position attention module"""
    #Ref from SAGAN
    def __init__(self, in_dim, height, width, cuda=False,):
        super(PAM_SE_Module, self).__init__()
        self.chanel_in = in_dim
        self.is_cuda = cuda

        self.query_conv = Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.key_conv = Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.value_conv = Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = Parameter(torch.zeros(1))

        self.softmax = Softmax(dim=-1)

        self.squeeze = Conv2d(in_channels=in_dim, out_channels=1, kernel_size=1)
        self.flatten = nn.Flatten()
        dim_se = height * width
        self.excitation = nn.Sequential(
            nn.Linear(in_features=dim_se, out_features=dim_se // 8),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=dim_se // 8, out_features=dim_se),
            nn.Sigmoid()
        )
    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X (HxW) X (HxW)
        """
        m_batchsize, C, height, width = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, width*height).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width*height)
        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width*height)

        pos_out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        pos_out = pos_out.view(m_batchsize, C, height, width)

        se_attention = self.squeeze(x)
        se_attention = self.flatten(se_attention)
        se_attention = self.excitation(se_attention)
        se_attention = se_attention.view(m_batchsize, height, width)

        if self.is_cuda == True:
            out = torch.zeros(pos_out.size()).cuda()
        else:
            out = torch.zeros(pos_out.size())

        for i in range(m_batchsize):
            out[i] = torch.mul(pos_out[i], se_attention[i])
        out = self.gamma*out + x
        return out

class CAM_SE_Module(Module):
    """ Channel attention module"""
    def __init__(self, in_dim, cuda=False):
        super(CAM_SE_Module, self).__init__()
        self.chanel_in = in_dim
        self.is_cuda = cuda

        self.gamma = Parameter(torch.zeros(1))
        self.softmax  = Softmax(dim=-1)

        self.squeeze = nn.AdaptiveAvgPool2d(1)
        # self.flatten = torch.flatten()
        self.flatten = nn.Flatten()
        #dim_se = height * width
        self.excitation = nn.Sequential(
            nn.Linear(in_features=in_dim, out_features=in_dim // 8),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=in_dim // 8, out_features=in_dim),
            nn.Sigmoid()
        )
    def forward(self,x):
        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X C X C
        """
        m_batchsize, C, height, width = x.size()
        proj_query = x.view(m_batchsize, C, -1)
        proj_key = x.view(m_batchsize, C, -1).permute(0, 2, 1)
        energy = torch.bmm(proj_query, proj_key)
        energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy)-energy
        attention = self.softmax(energy_new)
        proj_value = x.view(m_batchsize, C, -1)

        ch_out = torch.bmm(attention, proj_value)
        ch_out = ch_out.view(m_batchsize, C, height, width)

        se_attention = self.squeeze(x)
        se_attention = self.flatten(se_attention)
        se_attention = self.excitation(se_attention)
        #se_attention = se_attention.view(height, width)
        if self.is_cuda == True:
            out = torch.zeros(ch_out.size()).cuda()
        else:
            out = torch.zeros(ch_out.size())

        for i in range(m_batchsize):
            for j in range(C):
                out[i, j, :, :] = torch.mul(ch_out[i, j, :, :], se_attention[i, j])
        #out = torch.mul(ch_out, se_attention)
        out = self.gamma*out + x
        return out


if __name__ == "__main__":
    PA_model = PAM_SE_Module(in_dim=320, height=64, width=64)
    image = torch.randn(1, 320, 64, 64)
    out = PA_model(image)
    print(image.shape)
    print("input:", image.shape)
    #mini_mask, mask = model(image)
    #print("output:", mini_mask.shape)
    ##print('output:', mask.shape)
    print("output:", out.shape)