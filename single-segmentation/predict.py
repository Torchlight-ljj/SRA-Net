from dataset import ListDataset
import argparse
import torch
import torch.nn as nn
import numpy as np
from torch.utils.tensorboard import SummaryWriter   
import cv2
import time
import os
import model
from model.unet import UNet
from model.segnet import SegNet
from model.aunet import AUNet_Res
from model.fcn import FCN
from model.danet import DANet
from model.scse_unet import SCSERes
import torchvision.transforms as transforms
import random
import torch.nn.functional as F
import metrics 
from PIL import Image
def pad_to_square(img, pad_value):
    c, h, w = img.shape
    dim_diff = np.abs(h - w)
    # (upper / left) padding and (lower / right) padding
    pad1, pad2 = dim_diff // 2, dim_diff - dim_diff // 2
    # Determine padding
    pad = (0, 0, pad1, pad2) if h <= w else (pad1, pad2, 0, 0)
    # Add padding
    img = F.pad(img, pad, "constant", value=pad_value)

    return img, pad

def resize(image, size):
    image = F.interpolate(image.unsqueeze(0), size=size, mode="nearest").squeeze(0)
    return image

def test(ori_img, mask_img, save_path, weight_path, Net, line_color =(0,0,255), line_width = 5, add_mask= False):
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_size",type=int,default=512,help="")
    parser.add_argument("--class_num", type=int, default=2, help="")
    parser.add_argument("--weights_path", type=str, default=None, help="")
    opt = parser.parse_args()

    CLASS_NUM = opt.class_num
    WEIGHTS = weight_path
    INPUT_SIZE = opt.image_size

    SegNet = Net(n_channels=3, n_classes=CLASS_NUM, pretrained_model=False)
    dir_path =  save_path 
    if not os.path.isdir(dir_path):
        os.makedirs(dir_path)
    ori_image = cv2.imread(ori_img)
    if add_mask:
        mask = None
        classes = int(mask_img.split('/')[3])
        if classes==0:
            mask = np.zeros(ori_image.shape[:2])
        else:
            mask =(cv2.imread(mask_img,0))
            ori_image = cv2.resize(ori_image,(mask.shape[1],mask.shape[0]),cv2.INTER_CUBIC)
            contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(ori_image, contours, -1, line_color, line_width)

    image = Image.open(ori_img)
    # size = mask.shape
    image = transforms.ToTensor()(image)
    image = transforms.Resize((INPUT_SIZE,INPUT_SIZE))(image).unsqueeze(0).cuda()
    SegNet = SegNet.cuda()
    SegNet.load_state_dict(torch.load(WEIGHTS))
    c1 = 0
    c2 = 0
    c3 = 0
    with torch.no_grad():
        SegNet.eval()
        output = SegNet(image)
        output = output[0]
        _, predicted = torch.max(output.data, 0)
        predicted = predicted.cpu().numpy().astype(np.uint8)*255
        cv2.imwrite(os.path.join(dir_path,ori_img.split("/")[-1]),predicted)


nets = {
"unet":UNet(3,CLASS_NUM),
"segnet":SegNet(3,CLASS_NUM),
"aunet":AUNet_Res(3,CLASS_NUM,pretrained_model='./new_resnet50.pth'), \
"fcn":FCN(3,CLASS_NUM,pretrained_model='./new_resnet50.pth'),
"danet":DANet(CLASS_NUM,3,pretrained_model='./new_resnet50.pth'), \
"scsenet": SCSERes(3,CLASS_NUM,pretrained_model='./new_resnet50.pth')}
net = "aunet"
with open('predict.txt', "r") as file:
    img_files = file.readlines()
    random.shuffle(img_files)
    for name in img_files:
        name = name.rstrip()
        mask_name = name.replace("ori","mask").replace("jpg","png")
        clas = mask_name.split('/')[3]
        try:
            test(ori_img = name,mask_img = mask_name,save_path = './'+net+'/results/'+clas, weight_path='./'+net+'/save/1.pth', Net = nets[net], add_mask = False)
        except Exception as e:
            print("error:",name)
            continue
