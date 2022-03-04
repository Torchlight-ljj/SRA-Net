import torch
import torch.nn as nn
import numpy as np
from torch.utils.tensorboard import SummaryWriter   
import cv2
import time
import os
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import models
def count_params(model, input_size=512):
    # param_sum = 0
    with open('models.txt', 'w') as fm:
        fm.write(str(model))
 
    # 计算模型的计算量
    flops = calc_flops(model, input_size)
 
    # 计算模型的参数总量
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    return (params/1e6,flops)
    # print('The network has {} params.'.format(params/1e6))
 
 
# 计算模型的计算量
def calc_flops(model, input_size=512):
 
    def conv_hook(self, input, output):
        batch_size, input_channels, input_height, input_width = input[0].size()
        output_channels, output_height, output_width = output[0].size()
 
        kernel_ops = self.kernel_size[0] * self.kernel_size[1] * (self.in_channels / self.groups) * (
            2 if multiply_adds else 1)
        bias_ops = 1 if self.bias is not None else 0
 
        params = output_channels * (kernel_ops + bias_ops)
        flops = batch_size * params * output_height * output_width
 
        list_conv.append(flops)
 
    def linear_hook(self, input, output):
        batch_size = input[0].size(0) if input[0].dim() == 2 else 1
 
        weight_ops = self.weight.nelement() * (2 if multiply_adds else 1)
        bias_ops = self.bias.nelement()
 
        flops = batch_size * (weight_ops + bias_ops)
        list_linear.append(flops)
 
    def bn_hook(self, input, output):
        list_bn.append(input[0].nelement())
 
    def relu_hook(self, input, output):
        list_relu.append(input[0].nelement())
 
    def pooling_hook(self, input, output):
        batch_size, input_channels, input_height, input_width = input[0].size()
        output_channels, output_height, output_width = output[0].size()
 
        kernel_ops = self.kernel_size * self.kernel_size
        bias_ops = 0
        params = output_channels * (kernel_ops + bias_ops)
        flops = batch_size * params * output_height * output_width
 
        list_pooling.append(flops)
 
    def foo(net):
        childrens = list(net.children())
        if not childrens:
            if isinstance(net, torch.nn.Conv2d):
                net.register_forward_hook(conv_hook)
            if isinstance(net, torch.nn.Linear):
                net.register_forward_hook(linear_hook)
            if isinstance(net, torch.nn.BatchNorm2d):
                net.register_forward_hook(bn_hook)
            if isinstance(net, torch.nn.ReLU):
                net.register_forward_hook(relu_hook)
            if isinstance(net, torch.nn.MaxPool2d) or isinstance(net, torch.nn.AvgPool2d):
                net.register_forward_hook(pooling_hook)
            return
        for c in childrens:
            foo(c)
 
    multiply_adds = False
    list_conv, list_bn, list_relu, list_linear, list_pooling = [], [], [], [], []
    foo(model)
    if '0.4.' in torch.__version__:
        if assets.USE_GPU:
            input = torch.cuda.FloatTensor(torch.rand(2, 3, input_size, input_size).cuda())
        else:
            input = torch.FloatTensor(torch.rand(2, 3, input_size, input_size))
    else:
        input = Variable(torch.rand(2, 3, input_size, input_size), requires_grad=True)
    _ = model(input)
 
    total_flops = (sum(list_conv) + sum(list_linear) + sum(list_bn) + sum(list_relu) + sum(list_pooling))

    return (total_flops / 1e6 / 2)
 
    # print('  + Number of FLOPs: %.2fM' % (total_flops / 1e6 / 2))

# CLASS_NUM = 2
# nets = {"SRANet":UNet_Res(n_classes=CLASS_NUM,n_channels=3,pretrained_model_path=False),"segnet":SegNet(3,CLASS_NUM),"unet":UNet(n_classes=CLASS_NUM,n_channels=3), \
# "aunet":AUNet_Res(3,CLASS_NUM,pretrained_model='./new_resnet50.pth'),\
#         "fcn":FCN(3,CLASS_NUM,pretrained_model='./new_resnet50.pth'),"danet":DANet(CLASS_NUM,3,pretrained_model='./new_resnet50.pth'), \
#    "scsenet": SCSERes(3,CLASS_NUM,pretrained_model='./new_resnet50.pth')}
# class_net = {
# "resnet50": models.resnet50(pretrained = False, num_classes = 4),
# "densenet121": models.densenet121(pretrained = False, num_classes=4),
# "inception_v3": models.inception_v3(pretrained = False, aux_logits=False, num_classes = 4),
# "mobilenet_v2":models.mobilenet_v2(pretrained = False, num_classes = 4),
# "squeezenet1_1": models.squeezenet1_1(pretrained = False, num_classes = 4)
# }

# for net_name in nets.keys():
#     if net_name == "segnet":
#         nets[net_name].load_weights('vgg16_pre.pth')
#     paras,flops = count_params(nets[net_name],input_size=512)
#     # nParams = sum([p.nelement() for p in nets[net_name].parameters()])
#     print('*%s: number of parameters: %.3fM FLOPS: %.3fM.' % (net_name,round(paras,3),round(flops,3)))
# for net_name in class_net.keys():
#     paras,flops = count_params(class_net[net_name],input_size=512)
#     # nParams = sum([p.nelement() for p in nets[net_name].parameters()])
#     print('*%s: number of parameters: %.3fM FLOPS: %.3fM.' % (net_name,round(paras,3),round(flops,3)))