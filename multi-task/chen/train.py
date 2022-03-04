from dataset import ListDataset
import argparse
import torch
import torch.nn as nn
import numpy as np
from torch.utils.tensorboard import SummaryWriter   
import cv2
import time
import os
from model.unet import UNet
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


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.5, gamma=2, weight=None, ignore_index=255):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.weight = weight
        self.ignore_index = ignore_index
        self.ce_fn = nn.CrossEntropyLoss(weight=self.weight, ignore_index=self.ignore_index)
 
    def forward(self, preds, labels):
        logpt = -self.ce_fn(preds, labels)
        pt = torch.exp(logpt)
        loss = -((1 - pt) ** self.gamma) * self.alpha * logpt
        return loss
class MultiLoss(nn.Module):
    def __init__(self):
        super(MultiLoss, self).__init__()
        w = torch.Tensor([0.5,0.2])
        self.paras = (nn.Parameter(w)) 
    def forward(self,x1,x2):
        weight = torch.sigmoid(self.paras)
        y = weight[0]*x1 + weight[1]*x2 
        return y
def train(Net,MultiLoss,train_fold):
    writer = SummaryWriter(os.path.join(train_fold,"logs"))
    Net = Net.cuda()
    MultiLoss = MultiLoss.cuda()
    first_acc = 0
    acc = 0
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=BATCH_SIZE,shuffle=True,num_workers = 0)
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=1, shuffle=False,num_workers = 0)

    optimizer = torch.optim.SGD([{"params": Net.parameters()},{"params": MultiLoss.parameters()}], lr=LR, momentum=MOMENTUM)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.2)
    loss_func = nn.CrossEntropyLoss() 
    Net.train()
    glob_step = 0
    mIoU = []
    loss_cla = FocalLoss()
    for epoch in range(EPOCH):
        for step, (b_x, b_y, classes) in enumerate(train_loader):
            b_x = b_x.cuda()
            b_y = b_y.cuda()
            b_y = b_y.view(b_x.shape[0], INPUT_SIZE,INPUT_SIZE)
            classes = classes.cuda()
            output,y_cla = Net(b_x)
            loss = loss_func(output,b_y.long())
            loss1 = loss_cla(y_cla,classes)
            total_loss = MultiLoss(loss,loss1)
            total_loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            if step % 1 == 0:
                print("Epoch:{0} || Step:{1} || Loss:{2} || Loss1:{3} || w1:{4} || w2:{5}.".format(epoch, \
                    step, format(loss, ".4f"),format(loss1, ".4f"),format(float(MultiLoss.paras[0]), ".4f"),format(float(MultiLoss.paras[1]), ".4f")))
                writer.add_scalar('seg_loss', loss, glob_step)
                writer.add_scalar('cla_loss', loss1, glob_step)
                writer.add_scalar('w1', float(MultiLoss.paras[0]), glob_step)
                writer.add_scalar('w2', float(MultiLoss.paras[1]), glob_step)
                writer.add_scalar('lr',optimizer.state_dict()['param_groups'][0]['lr'],glob_step)
            glob_step += 1
        scheduler.step()
        if epoch % 1 == 0:
            correct_num = 0
            DSC = {'total_mean':[],'0_class':[],'1_class':[],'2_class':[],'3_class':[]}
            GT = []
            Pred = []
            IOU_s =[]
            FPS = []
            for step,(b_x,b_y,cla_true) in enumerate(val_loader):
                b_x = b_x.cuda()
                b_y = b_y.cpu()
                b_y = b_y.view(b_x.shape[0], INPUT_SIZE,INPUT_SIZE).int()
                cla_true = cla_true.cuda().squeeze()
                with torch.no_grad():
                    Net.eval()
                    start = time.time()
                    output,y_cla = Net(b_x)
                    end = time.time()
                    FPS.append(1/(end-start)) 
                    predicted = torch.argmax(output.data, 1).cpu()
                    y_cla = y_cla.squeeze()
                    y_cla = torch.argmax(y_cla,0)
                    predicted = torch.argmax(output.data, 1).cpu()
                    iou = metrics.compute_iou(predicted.cpu().numpy(),b_y.cpu().numpy())
                    IOU_s.append(iou)
                    b_y = metrics.get_onehot(b_y,CLASS_NUM,INPUT_SIZE,INPUT_SIZE)
                    predicted = metrics.get_onehot(predicted.int(),CLASS_NUM,INPUT_SIZE,INPUT_SIZE)
                    if b_y.dim() == 3:
                        predicted = predicted.unsqueeze(0)
                        b_y = b_y.unsqueeze(0)
                    dice,dice_class = metrics.multiclass_dice_coeff(predicted,b_y)
                    GT.append(cla_true.cpu().numpy())
                    Pred.append(y_cla.cpu().numpy())
                    DSC['total_mean'].append(dice)

            dice_arr = DSC['total_mean']
            FPS = np.array(FPS)
            print('dice:{:.3f},std:{:.3f},miou:{:.3f},std:{:.3f},fps:{:.3f}'.format(np.mean(dice_arr),np.std(dice_arr),np.mean(IOU_s),np.std(IOU_s),FPS.mean()))
            y_true = np.array(GT)
            y_pred = np.array(Pred)
            print(y_pred.shape,y_true.shape)

            classify_result = metrics.compute_classify(y_pred,y_true)
            print('Acc:{:.3f},Precision:{:.3f},Recall:{:.3f},F1:{:.3f}'.format(classify_result[0],classify_result[1],classify_result[2],classify_result[3]))
            file = open(os.path.join(train_fold,"result.txt"), "a") 
            file.write("evalution time: " + str(time.asctime(time.localtime(time.time()))) + "\n")
            file.write('dice:{:.3f},std:{:.3f},miou:{:.3f},std:{:.3f},fps:{:.3f},epoch:{:3d}.\n'.format(np.mean(dice_arr),np.std(dice_arr),np.mean(IOU_s),np.std(IOU_s),FPS.mean(),epoch))
            file.write('Acc:{:.3f},Precision:{:.3f},Recall:{:.3f},F1:{:.3f}\n'.format(classify_result[0],classify_result[1],classify_result[2],classify_result[3]))
            file.close()  
        torch.save(Net.state_dict(), train_fold+"/save/"+ str(epoch) + ".pth")
        torch.save(MultiLoss.state_dict(), train_fold+"/save/"+ str(epoch) + "_weights"+ ".pth")


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

opt = parser.parse_args()
print(opt)

CLASS_NUM = opt.class_num
EPOCH = opt.epoch
BATCH_SIZE = opt.batch_size
LR = opt.learning_rate
MOMENTUM = opt.momentum
CATE_WEIGHT = opt.category_weight
TXT_PATH = opt.train_txt
INPUT_SIZE = opt.image_size
VAL_PATH = opt.val_txt
transform_train = transforms.Compose([
    transforms.Resize((INPUT_SIZE,INPUT_SIZE)),
    transforms.RandomRotation((0,60)),
    transforms.RandomAffine(10,),
    transforms.RandomHorizontalFlip(), 
])
train_data = ListDataset(TXT_PATH,image_size=INPUT_SIZE,train=True,is_ours=True)
val_data = ListDataset(VAL_PATH,image_size=INPUT_SIZE,train=False,is_ours=True)
train_fold ="data"
if not os.path.exists(train_fold):
    os.mkdir('./'+ train_fold)
    os.mkdir('./'+train_fold+'/save')
    os.mkdir('./'+train_fold+'/figs')
Net = UNet(n_channels= 3, n_classes=CLASS_NUM, cla_n_classes=4)
train(Net,MultiLoss(), train_fold)