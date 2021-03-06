import io
import requests
from PIL import Image
import torch
from torchvision import transforms
from torchvision import models
from torch.autograd import Variable
from torch.nn import functional as F
import numpy as np
import cv2
import json
import os
from model.sranet import UNet_Res

class Cam():
    def __init__(self, Net, weight_path, input_size,  cuda_flag = True):
        super(Cam, self).__init__()
        self.cuda_flag = cuda_flag
        if self.cuda_flag:
            self.Net = Net.cuda()
        else:
            self.Net = Net
        try:
            self.Net.load_state_dict(torch.load(weight_path))
        except Exception as e:
            raise e
        self.input_size = input_size
        self.fmap_block = list()
        self.grad_block = list()
        self.ori_shape = None

    def get_cam(self, img_path, out_dir):
        img_pil = Image.open(img_path)
        self.ori_shape = img_pil.size
        ori = transforms.ToTensor()(img_pil)
        ori = transforms.Resize((self.input_size, self.input_size))(ori)
        if self.cuda_flag:
            img_variable = ori.unsqueeze(0).cuda()
        else:
            img_variable = ori.unsqueeze(0)
        #model
        self.Net.eval()
        classes = list(key for key in range(4))
        # regist the hook
        # print(self.Net._modules)
        self.Net.context_path.layer4.register_forward_hook(self.farward_hook)
        self.Net.context_path.layer4.register_backward_hook(self.backward_hook)

        # forward
        seg, output, features = self.Net(img_variable)
        idx = np.argmax(output.cpu().data.numpy(),axis=1)[0]
        print("predict: {}".format(classes[idx]))

        # backward
        self.Net.zero_grad()
        class_loss = output[0,idx]
        class_loss.backward()

        # generate cam
        grads_val = self.grad_block[0].cpu().data.numpy().squeeze()
        fmap = self.fmap_block[0].cpu().data.numpy().squeeze()

        # save cam
        self.cam_show_img(img_path, fmap, grads_val, out_dir)

    def backward_hook(self, module, grad_in, grad_out):
        self.grad_block.append(grad_out[0].detach())

    def farward_hook(self, module, input, output):
        self.fmap_block.append(output)

    def cam_show_img(self, img_path, feature_map, grads, out_dir):
        img = cv2.imread(img_path)
        # H, W, _ = img.shape
        cam = np.zeros(feature_map.shape[1:], dtype=np.float32)		# 4
        grads = grads.reshape([grads.shape[0],-1])					# 5
        weights = np.mean(grads, axis=1)							# 6
        for i, w in enumerate(weights):
            cam += w * feature_map[i, :, :]							# 7
        cam = np.maximum(cam, 0)
        cam = cam / cam.max()
        cam = cv2.resize(cam, (512, 512))

        heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
        img = cv2.resize(img, (512, 512))
        path_cam_img = out_dir
        cv2.imwrite(path_cam_img, heatmap)
        mask_path= img_path.replace("jpg","png").replace("ori","mask")
        mask = cv2.imread(mask_path,0)
        mask = cv2.resize(mask, (512, 512))
        cv2.imwrite(os.path.join('./grad_cam/ori',path_cam_img.split('/')[-1]),img)
        cv2.imwrite(os.path.join('./grad_cam/mask',path_cam_img.split('/')[-1]),mask)

Net = UNet_Res(3,2)
# print(Net)
img_list =[
'./data/ori/1/?????????/?????????_???_61???_06716272_?????????_EUS62892-0120211025101959007.jpg',
'./data/ori/2/??????/??????_???_36???_06609121_?????????_EUS63598-0120211011150926063.jpg',
'./data/ori/3/?????????/?????????_???_67???_QG12027610_?????????_EUS63878-0120211021125432078.jpg',
'./data/ori/3/??????/??????_???_53???_06650440_?????????_EUS63312-0120211021115932033.jpg',
]
for img in img_list:
    CAM = Cam(Net,'./save/6.pth',512,True)
    CAM.get_cam(img, './grad_cam/' + img.split('/')[-1])