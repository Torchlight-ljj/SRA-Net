from dataset import ListDataset
import argparse
import torch
import torch.nn as nn
import numpy as np
from torch.utils.tensorboard import SummaryWriter   
import cv2
import time
import os
from model.sranet import UNet_Res
import torchvision.transforms as transforms
import random
import torch.nn.functional as F
import metrics
def setup_seed(seed):
    #固定一个随机因子
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

def get_att_dis(target, behaviored):
    attention_distribution = []
    for i in range(behaviored.size(0)):
        attention_score = torch.cosine_similarity(target, behaviored[i].view(1, -1))  # 计算每一个元素与给定元素的余弦相似度
        attention_distribution.append(attention_score)
    attention_distribution = torch.Tensor(attention_distribution)

    return attention_distribution
def simi(z,label,class_nums):
    #input:batch*channel
    #label:batch*1
    batch_size = z.shape[0]
    sort = list(label.cpu().numpy().astype(int))
    y = {}
    for i in range(class_nums):
        y.setdefault(str(i),[])
    # y = {"0":[],"1":[],"2":[],"3":[]}
    for i in range(batch_size):
        y[str(sort[i])].append(i)
    class_inter = torch.Tensor([0]).cuda()
    class_outer = torch.Tensor([0]).cuda()
    class_indexes = []
    for key in y.keys():
        idx = y[key]
        if len(idx) == 2:
            class_inter += torch.cosine_similarity(z[idx[0]], z[idx[1]], dim=0)
        if len(idx) == 1:
            class_inter += torch.Tensor([1]).cuda()
        if len(idx) > 2:
            cat_M = []
            for i in range(1,len(idx)):
                cat_M.append(z[idx[i]].unsqueeze(0))
                # print(z[idx[i]].unsqueeze(0).shape)
            cat_M = torch.cat(cat_M, dim=0)
            class_inter += get_att_dis(z[idx[0]].unsqueeze(0),cat_M).mean()
        if len(idx) > 0:
            class_indexes.append(key)
        
    if len(class_indexes) > 1:
        classes_out = []
        for index in class_indexes:
            classes_out.append(z[y[index][0]].unsqueeze(0))
        classes_outs = torch.cat(classes_out[1:], dim=0)
        class_outer += get_att_dis(classes_out[0],classes_outs).sum()
        
    return (class_outer-class_inter)/len(class_indexes)

class MultiLoss(nn.Module):
    def __init__(self):
        super(MultiLoss, self).__init__()
        w = torch.Tensor([0.5,0.2,0.3])
        self.paras = (nn.Parameter(w)) 
    def forward(self,x1,x2,x3):
        weight = torch.sigmoid(self.paras)
        y = weight[0]*x1 + weight[1]*x2 + weight[2]*x3
        return y
def train(SegNet,MultiLoss,train_fold):
    writer = SummaryWriter(os.path.join(train_fold,"logs"))
    SegNet = SegNet.cuda()
    MultiLoss = MultiLoss.cuda()
    # SegNet.load_weights(PRE_TRAINING)
    first_acc = 0
    acc = 0
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=BATCH_SIZE,shuffle=True,num_workers = 0)
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=1, shuffle=False,num_workers = 0)

    optimizer = torch.optim.SGD([{"params": SegNet.parameters()},{"params": MultiLoss.parameters()}], lr=LR, momentum=MOMENTUM)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.2)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(train_loader), eta_min=0,last_epoch=-1)
    # loss_func = nn.CrossEntropyLoss(weight=torch.from_numpy(np.array(CATE_WEIGHT)).float()).cuda()
    loss_func = nn.CrossEntropyLoss() 
    SegNet.train()
    glob_step = 0
    mIoU = []
    loss_cla = FocalLoss()
    for epoch in range(EPOCH):
        for step, (b_x, b_y, classes) in enumerate(train_loader):
            b_x = b_x.cuda()
            b_y = b_y.cuda()
            b_y = b_y.view(b_x.shape[0], INPUT_SIZE,INPUT_SIZE)
            classes = classes.cuda()
            output,y_cla,features = SegNet(b_x)
            features = features.cuda()
            loss = loss_func(output,b_y.long())
            loss1 = loss_cla(y_cla,classes)
            loss2 = simi(features,classes,4)
            # loss2 = 0
            total_loss = MultiLoss(loss,loss1,loss2)

            total_loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        
            if step % 1 == 0:
                print("Epoch:{0} || Step:{1} || Loss:{2} || Loss1:{3} || Loss2:{4} || w1:{5} || w2:{6} || w3:{7}".format(epoch, \
                    step, format(loss, ".4f"),format(loss1, ".4f"),format(float(loss2), ".4f"),format(float(MultiLoss.paras[0]), ".4f"),format(float(MultiLoss.paras[1]), ".4f"),format(float(MultiLoss.paras[2]), ".4f")))
                writer.add_scalar('seg_loss', loss, glob_step)
                writer.add_scalar('cla_loss', loss1, glob_step)
                writer.add_scalar('sim_loss', loss2, glob_step)
                writer.add_scalar('w1', float(MultiLoss.paras[0]), glob_step)
                writer.add_scalar('w2', float(MultiLoss.paras[1]), glob_step)
                writer.add_scalar('w3', float(MultiLoss.paras[2]), glob_step)
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
                classes = classes.cuda().squeeze()
                with torch.no_grad():
                    SegNet.eval()
                    output,y_cla,fea = SegNet(b_x)
                    y_cla = y_cla.squeeze()
                    y_cla = torch.argmax(y_cla,0)
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
                    if classes == y_cla:
                        correct_num += 1
                # intersection = []
                # union = []
                # iou = 0
                # # for i in range(1, CLASS_NUM):
                # #     intersection.append(np.sum(predict[target == i] == i))
                # #     union.append(np.sum(predict == i) + np.sum(target == i) - intersection[i-1])
                # #     iou += intersection[i-1]/union[i-1]
                # # 用numpy库实现的方法
                # intersection = np.logical_and(b_y, predicted)
                # union = np.logical_or(b_y, predicted)
                # iou = np.sum(intersection) / np.sum(union)
                # mIoU.append(iou) 
            mean_DSC = np.mean(DSC['total_mean'])
            print('DSC:{:.3f},mIoU:{:.3f},epoch:{:3d}.'.format(mean_DSC,mean_DSC/(2-mean_DSC),epoch))

            cla_acc = correct_num/len(val_loader)
            file = open(os.path.join(train_fold,"result.txt"), "a")            
            file.write("评价日期：" + str(time.asctime(time.localtime(time.time()))) + "\n")
            file.write('DSC:{:.3f},mIoU:{:.3f},class_acc:{:.3f},epoch:{:3d}.\n'.format(mean_DSC,mean_DSC/(2-mean_DSC),cla_acc,epoch))
            file.close()  
        torch.save(SegNet.state_dict(), train_fold+"/save/"+ str(epoch) + ".pth")
        torch.save(MultiLoss.state_dict(), train_fold+"/save/"+ str(epoch) + "_weights"+ ".pth")


parser = argparse.ArgumentParser()
parser.add_argument("--image_size",type=int,default=512,help="")
parser.add_argument("--class_num", type=int, default=2, help="")
parser.add_argument("--epoch", type=int, default=20, help="")
parser.add_argument("--batch_size", type=int, default=8, help="")
parser.add_argument("--learning_rate", type=float, default=0.005, help="")
parser.add_argument("--momentum", type=float, default=0.9)
parser.add_argument("--category_weight", type=float, default=[0.7502381287857225, 1.4990483912788268,1], help="")
parser.add_argument("--train_txt", type=str, default="cat.txt", help="")
parser.add_argument("--val_txt", type=str, default="val.txt", help="")

parser.add_argument("--pre_training_weight", type=str, default="resnet50.pth", help="")
parser.add_argument("--weights", type=str, default="./weights/", help="")
parser.add_argument("--val_paths", type=str, default="val.txt", help="验证集的图片和标签的路径")

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
train_data = ListDataset(TXT_PATH,image_size=INPUT_SIZE,train=True,is_ours=True)
val_data = ListDataset(VAL_PATH,image_size=INPUT_SIZE,train=False,is_ours=True)
# SegNet = SegNet(3, CLASS_NUM)
# train_fold = time.strftime("%Y-%m-%d %X", time.localtime())
train_fold ="SRA-Net"
if not os.path.exists(train_fold):
    os.mkdir('./'+ train_fold)
    os.mkdir('./'+train_fold+'/save')
    os.mkdir('./'+train_fold+'/figs')
SegNet = UNet_Res(3, CLASS_NUM, pretrained_model_path='./new_resnet50.pth')
train(SegNet,MultiLoss(), train_fold)
