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
from model.vgg import vgg16
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
        w = torch.Tensor([0.5,0.2])
        self.paras = (nn.Parameter(w)) 
    def forward(self,x1,x2):
        weight = torch.sigmoid(self.paras)
        y = weight[0]*x1 + weight[1]*x2 
        return y
def train(SegNet,train_fold,aux_model):
    writer = SummaryWriter(os.path.join(train_fold,"logs"))
    SegNet = SegNet.cuda()
    vgg_aux = aux_model.cuda()
    # MultiLoss = MultiLoss.cuda()
    # SegNet.load_weights(PRE_TRAINING)
    first_acc = 0
    acc = 0
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=BATCH_SIZE,shuffle=True,num_workers = 0)
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=1, shuffle=False,num_workers = 0)

    optimizer = torch.optim.SGD([{"params": SegNet.parameters()},{"params": vgg_aux.parameters()}], lr=LR, momentum=MOMENTUM)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.2)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(train_loader), eta_min=0,last_epoch=-1)
    # loss_func = nn.CrossEntropyLoss(weight=torch.from_numpy(np.array(CATE_WEIGHT)).float()).cuda()
    loss_func = nn.CrossEntropyLoss()
    loss_func1 = nn.MSELoss()  
    SegNet.train()
    vgg_aux.train()
    glob_step = 0
    mIoU = []
    loss_cla = FocalLoss()
    for epoch in range(EPOCH):
        for step, (b_x, b_y, classes) in enumerate(train_loader):
            b_x = b_x.cuda()
            b_y = b_y.cuda()
            b_y = b_y.view(b_x.shape[0], INPUT_SIZE,INPUT_SIZE)
            classes = classes.cuda()
            output = SegNet(b_x)
            vgg_input_preds = torch.argmax(output.data, 1).float()
            # vgg_input_preds = metrics.get_onehot(vgg_input_preds.cpu().int(),CLASS_NUM,INPUT_SIZE,INPUT_SIZE).cuda()
            # vgg_input_mask = metrics.get_onehot(b_y.cpu().int(),CLASS_NUM,INPUT_SIZE,INPUT_SIZE).cuda()
            L_list1 = vgg_aux(b_y.unsqueeze(1))
            L_list2 = vgg_aux(vgg_input_preds.unsqueeze(1))
            loss = loss_func(output,b_y.long())
            loss1 = loss_func1(L_list1[0],L_list2[0]) + loss_func1(L_list1[1],L_list2[1])+loss_func1(L_list1[2],L_list2[2])+loss_func1(L_list1[3],L_list2[3])
            loss1 = torch.div(loss1,4)
            total_loss = loss + 0.1*loss1
            total_loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        
            if step % 1 == 0:
                print("Epoch:{0} || Step:{1} || Loss:{2} || Loss1:{3}.".format(epoch, \
                    step, format(loss, ".4f"),format(loss1, ".4f")))
                writer.add_scalar('seg_loss', loss, glob_step)
                writer.add_scalar('vgg_loss', loss1, glob_step)
                # writer.add_scalar('w1', float(MultiLoss.paras[0]), glob_step)
                # writer.add_scalar('w2', float(MultiLoss.paras[1]), glob_step)
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
                # cla_true = cla_true.cuda().squeeze()
                with torch.no_grad():
                    SegNet.eval()
                    start = time.time()
                    output = SegNet(b_x)
                    end = time.time()
                    FPS.append(1/(end-start)) 
                    # predicted = torch.argmax(output.data, 1).cpu()
                    # y_cla = y_cla.squeeze()
                    # y_cla = torch.argmax(y_cla,0)
                    predicted = torch.argmax(output.data, 1).cpu()
                    iou = metrics.compute_iou(predicted.cpu().numpy(),b_y.cpu().numpy())
                    IOU_s.append(iou)
                    b_y = metrics.get_onehot(b_y,CLASS_NUM,INPUT_SIZE,INPUT_SIZE)
                    predicted = metrics.get_onehot(predicted.int(),CLASS_NUM,INPUT_SIZE,INPUT_SIZE)
                    if b_y.dim() == 3:
                        predicted = predicted.unsqueeze(0)
                        b_y = b_y.unsqueeze(0)
                    dice,dice_class = metrics.multiclass_dice_coeff(predicted,b_y)
                    # GT.append(cla_true.cpu().numpy())
                    # Pred.append(y_cla.cpu().numpy())
                    DSC['total_mean'].append(dice)

            dice_arr = DSC['total_mean']
            FPS = np.array(FPS)
            print('dice:{:.3f},std:{:.3f},miou:{:.3f},std:{:.3f},fps:{:.3f}'.format(np.mean(dice_arr),np.std(dice_arr),np.mean(IOU_s),np.std(IOU_s),FPS.mean()))
            # y_true = np.array(GT)
            # y_pred = np.array(Pred)
            # print(y_pred.shape,y_true.shape)

            # classify_result = metrics.compute_classify(y_pred,y_true)
            # print('Acc:{:.3f},Precision:{:.3f},Recall:{:.3f},F1:{:.3f}'.format(classify_result[0],classify_result[1],classify_result[2],classify_result[3]))
            file = open(os.path.join(train_fold,"result.txt"), "a") 
            file.write("评价日期：" + str(time.asctime(time.localtime(time.time()))) + "\n")
            file.write('dice:{:.3f},std:{:.3f},miou:{:.3f},std:{:.3f},fps:{:.3f},epoch:{:3d}.\n'.format(np.mean(dice_arr),np.std(dice_arr),np.mean(IOU_s),np.std(IOU_s),FPS.mean(),epoch))
            # file.write('Acc:{:.3f},Precision:{:.3f},Recall:{:.3f},F1:{:.3f}\n'.format(classify_result[0],classify_result[1],classify_result[2],classify_result[3]))
            file.close()  
        torch.save(SegNet.state_dict(), train_fold+"/save/"+ str(epoch) + ".pth")
        # torch.save(MultiLoss.state_dict(), train_fold+"/save/"+ str(epoch) + "_weights"+ ".pth")


parser = argparse.ArgumentParser()
parser.add_argument("--image_size",type=int,default=512,help="")
parser.add_argument("--class_num", type=int, default=4, help="")
parser.add_argument("--epoch", type=int, default=10, help="")
parser.add_argument("--batch_size", type=int, default=8, help="")
parser.add_argument("--learning_rate", type=float, default=0.005, help="")
parser.add_argument("--momentum", type=float, default=0.9)
parser.add_argument("--category_weight", type=float, default=[0.7502381287857225, 1.4990483912788268,1], help="")
parser.add_argument("--train_txt", type=str, default="train.txt", help="")
parser.add_argument("--val_txt", type=str, default="val.txt", help="")

parser.add_argument("--pre_training_weight", type=str, default="resnet50.pth", help="")
parser.add_argument("--weights", type=str, default="./weights/", help="")

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
train_data = ListDataset(TXT_PATH,image_size=INPUT_SIZE,train=True,is_ours=False)
val_data = ListDataset(VAL_PATH,image_size=INPUT_SIZE,train=False,is_ours=False)
# SegNet = SegNet(3, CLASS_NUM)
# train_fold = time.strftime("%Y-%m-%d %X", time.localtime())
train_fold ="data"
if not os.path.exists(train_fold):
    os.mkdir('./'+ train_fold)
    os.mkdir('./'+train_fold+'/save')
    os.mkdir('./'+train_fold+'/figs')
SegNet = UNet_Res(n_channels= 3, n_classes=CLASS_NUM)
Branch_vgg = vgg16(pretrained=False)
Branch_vgg.load_state_dict(torch.load('./vgg16.pth'),False)
train(SegNet,train_fold,Branch_vgg)
