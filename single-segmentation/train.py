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

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
setup_seed(20)

def train(Net,train_fold):
    writer = SummaryWriter(os.path.join(train_fold,"logs"))
    Net = Net.cuda()
    first_acc = 0
    acc = 0
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=BATCH_SIZE,shuffle=True,num_workers = 0)
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=1, shuffle=False,num_workers = 0)

    optimizer = torch.optim.SGD(Net.parameters(), lr=LR, momentum=MOMENTUM)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.2)
    loss_func = nn.CrossEntropyLoss() 
    Net.train()
    glob_step = 0
    mIoU = []
    for epoch in range(EPOCH):
        for step, (b_x, b_y, classes) in enumerate(train_loader):
            b_x = b_x.cuda()
            b_y = b_y.cuda()
            b_y = b_y.view(b_x.shape[0], INPUT_SIZE,INPUT_SIZE)
            output = Net(b_x)
            loss = loss_func(output,b_y.long())
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            if step % 1 == 0:
                print("Epoch:{0} || Step:{1}/{2} || Loss:{3}".format(epoch, step,len(train_loader), format(loss, ".4f")))
                writer.add_scalar('loss', loss, glob_step)
                writer.add_scalar('lr',optimizer.state_dict()['param_groups'][0]['lr'],glob_step)
            glob_step += 1

        scheduler.step()
        if epoch % 1 == 0:
            correct_num = 0
            DSC = {'total_mean':[],'0_class':[],'1_class':[],'2_class':[],'3_class':[]}
            for step,(b_x,b_y,classes) in enumerate(val_loader):
                b_x = b_x.cuda()
                b_y = b_y.cpu()
                b_y = b_y.view(b_x.shape[0], INPUT_SIZE,INPUT_SIZE).int()
                with torch.no_grad():
                    Net.eval()
                    output = Net(b_x)
                    predicted = torch.argmax(output.data, 1).cpu()
                    
                    b_y =metrics.get_onehot(b_y,CLASS_NUM,INPUT_SIZE,INPUT_SIZE)
                    predicted = metrics.get_onehot(predicted.int(),CLASS_NUM,INPUT_SIZE,INPUT_SIZE)
                    if b_y.dim() == 3:
                        predicted = predicted.unsqueeze(0)
                        b_y = b_y.unsqueeze(0)                    
                    dice,dice_class = metrics.multiclass_dice_coeff(predicted,b_y)
                    DSC['total_mean'].append(dice)
                    DSC['0_class'].append(dice_class[0])
                    DSC['1_class'].append(dice_class[1])  
            mean_DSC = np.mean(DSC['total_mean'])
            print('DSC:{:.3f},mIoU:{:.3f},epoch:{:3d}.'.format(mean_DSC,mean_DSC/(2-mean_DSC),epoch))
            file = open(os.path.join(train_fold,"result.txt"), "a")            
            file.write("evaluating time:" + str(time.asctime(time.localtime(time.time()))) + "\n")
            file.write('DSC:{:.3f},mIoU:{:.3f},epoch:{:3d}.\n'.format(mean_DSC,mean_DSC/(2-mean_DSC),epoch))
            file.close()  
        torch.save(Net.state_dict(), train_fold+"/save/"+ str(epoch) + ".pth")

parser = argparse.ArgumentParser()
parser.add_argument("--image_size",type=int,default=512,help="")
parser.add_argument("--class_num", type=int, default=2, help="")
parser.add_argument("--epoch", type=int, default=10, help="")
parser.add_argument("--batch_size", type=int, default=8, help="")
parser.add_argument("--learning_rate", type=float, default=0.005, help="")
parser.add_argument("--momentum", type=float, default=0.9)
parser.add_argument("--category_weight", type=float, default=[0.7502381287857225, 1.4990483912788268,1], help="")
parser.add_argument("--train_txt", type=str, default="train.txt", help="")
parser.add_argument("--val_txt", type=str, default="val.txt", help="")
parser.add_argument("--pre_training_weight", type=str, default="resnet50.pth", help="")
parser.add_argument("--weights", type=str, default="./weights/", help="")
parser.add_argument("--val_paths", type=str, default="val.txt", help="")
opt = parser.parse_args()
print(opt)
CLASS_NUM = opt.class_num
EPOCH = opt.epoch
BATCH_SIZE = opt.batch_size
LR = opt.learning_rate
MOMENTUM = opt.momentum
CATE_WEIGHT = opt.category_weight
TXT_PATH = opt.train_txt
PRE_TRAINING = opt.pre_training_weight
WEIGHTS = opt.weights
INPUT_SIZE = opt.image_size
VAL_PATH = opt.val_txt
transform_train = transforms.Compose([
    transforms.Resize((INPUT_SIZE,INPUT_SIZE)),
    transforms.RandomRotation((0,60)),
    transforms.RandomAffine(10,),
    transforms.RandomHorizontalFlip(), 
])
train_data = ListDataset(TXT_PATH,image_size=INPUT_SIZE,train=True, is_ours=True)
val_data = ListDataset(VAL_PATH,image_size=INPUT_SIZE,train=False, is_ours=True)
nets = {
"unet":UNet(3,CLASS_NUM),
"segnet":SegNet(3,CLASS_NUM),
"aunet":AUNet_Res(3,CLASS_NUM,pretrained_model='./new_resnet50.pth'), \
"fcn":FCN(3,CLASS_NUM,pretrained_model='./new_resnet50.pth'),
"danet":DANet(CLASS_NUM,3,pretrained_model='./new_resnet50.pth'), \
"scsenet": SCSERes(3,CLASS_NUM,pretrained_model='./new_resnet50.pth')}
for net_name in nets.keys():
    if not os.path.exists(net_name):
        os.mkdir('./'+ net_name)
        os.mkdir('./'+net_name+'/save')
        os.mkdir('./'+net_name+'/figs')
    if net_name == "segnet":
        nets[net_name].load_weights('vgg16_pre.pth')
    train(nets[net_name],net_name)

