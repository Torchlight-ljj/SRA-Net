from dataset import ListDataset
import argparse
import torch
import torch.nn as nn
import numpy as np
from torch.utils.tensorboard import SummaryWriter   
import cv2
import time
import os
from sklearn.metrics import accuracy_score, average_precision_score, precision_score,f1_score,recall_score

from model.sranet import UNet_Res
import metrics
from torch import Tensor
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('agg')
from sklearn.metrics import confusion_matrix
def plot_matrix(matrix, save_name, labels_name, title=None, thresh=15000, axis_labels=None):
    cm = matrix.astype('float') / matrix.sum(axis=1)[:, np.newaxis]  # 归一化
    print(cm.shape)
    cm = matrix.astype('float')
    print(cm.shape)
    # cm = matrix
# 画图，如果希望改变颜色风格，可以改变此部分的cmap=pl.get_cmap('Blues')处
    plt.imshow(cm, interpolation='nearest', cmap=plt.get_cmap('Reds'))
    plt.colorbar()  # 绘制图例

# 图像标题
    if title is not None:
        plt.title(title)
# 绘制坐标
    num_local = np.array(range(len(labels_name)))
    if axis_labels is None:
        axis_labels = labels_name
    plt.xticks(num_local, axis_labels)  # 将标签印在x轴坐标上， 并倾斜45度
    plt.yticks(num_local, axis_labels)  # 将标签印在y轴坐标上
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

# 将百分比打印在相应的格子内，大于thresh的用白字，小于的用黑字
    for i in range(np.shape(cm)[0]):
        for j in range(np.shape(cm)[1]):
            if int(cm[i][j]) > 0:
                plt.text(j, i, format(int(cm[i][j]), 'd'),
                        ha="center", va="center",
                        color="white" if cm[i][j] > thresh else "black") 
    plt.savefig(save_name, transparent=True, dpi=800)   
    plt.close()
def get_onehot(label:Tensor, num_class, high, width):
    if label.shape[0] == 1:
        label = label.view(-1)
        ones = torch.sparse.torch.eye(num_class)
        ones = ones.index_select(0, label)
        ones = torch.transpose(ones, 0, 1).int()
        ones_high = ones.shape[0]
        label_list = []
        for i in range(ones_high):
            label_list.append(ones[i].reshape(high, width).unsqueeze(0))
        return torch.cat(label_list,dim=0)
    else:
        batch_label = []
        for batch in range(label.shape[0]):
            label_tem = label[batch].view(-1)
            ones = torch.sparse.torch.eye(num_class)
            ones = ones.index_select(0, label_tem)
            ones = torch.transpose(ones, 0, 1).int()
            ones_high = ones.shape[0]
            label_list = []
            for i in range(ones_high):
                label_list.append(ones[i].reshape(high, width).unsqueeze(0))
            batch_label.append(torch.cat(label_list,dim=0).unsqueeze(0))
        return torch.cat(batch_label,dim=0)

def iou_score(output, target):
    smooth = 1e-5

    if torch.is_tensor(output):
        output = torch.sigmoid(output).data.cpu().numpy()
    if torch.is_tensor(target):
        target = target.data.cpu().numpy()
    output_ = output > 0.5
    target_ = target > 0.5
    intersection = (output_ & target_).sum()
    union = (output_ | target_).sum()

    return (intersection + smooth) / (union + smooth)

parser = argparse.ArgumentParser()
parser.add_argument("--image_size",type=int,default=512,help="")
parser.add_argument("--class_num", type=int, default=4, help="")
parser.add_argument("--batch_size", type=int, default=1, help="")
parser.add_argument("--val_txt", type=str, default="val.txt", help="")

opt = parser.parse_args()
print(opt)

CLASS_NUM = opt.class_num
BATCH_SIZE = opt.batch_size
INPUT_SIZE = opt.image_size
VAL_PATH = opt.val_txt
val_data = ListDataset(opt.val_txt,image_size=INPUT_SIZE,transform=None,train=False,is_ours=False)

SegNet = UNet_Res(3, CLASS_NUM, pretrained_model_path=None)
SegNet = SegNet.cuda()
SegNet.load_state_dict(torch.load('./data/save/6.pth'))
DSC = {'total_mean':[],'0_class':[],'1_class':[],'2_class':[],'3_class':[]}

IOU_s =[]
val_loader = torch.utils.data.DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=False,num_workers = 0)
nums = 0
GT = []
Pred = []

seg_gt = []
seg_pred = []

seg_prec = []
seg_recal = []
FPS = []
for step,(b_x,b_y,cla_true) in enumerate(val_loader):
    b_x = b_x.cuda()
    b_y = b_y.cpu()
    b_y = b_y.view(b_x.shape[0], INPUT_SIZE,INPUT_SIZE).int()
    cla_true = cla_true.squeeze()
    with torch.no_grad():
        SegNet.eval()
        start = time.time()
        output = SegNet(b_x)
        end = time.time()
        FPS.append(1/(end-start)) 
        predicted = torch.argmax(output.data, 1).cpu()

        if predicted.max()==0:
            y_cla = 0
        else:
            nums = torch.Tensor([(predicted==1).sum(),(predicted==2).sum(),(predicted==3).sum()])
            y_cla = torch.argmax(nums,dim=0) + 1

        predicted = (predicted>=1)*1
        b_y = (b_y>=1)*1

        # print(predicted.shape,b_y.shape)
        # while True:
        #     pass
        # y_cla = y_cla.squeeze()
        # y_cla = torch.argmax(y_cla,0)
        # predicted = torch.argmax(output.data, 1).cpu()

        iou = metrics.compute_iou(predicted.cpu().numpy(),b_y.cpu().numpy())
        IOU_s.append(iou)
        b_y = get_onehot(b_y,CLASS_NUM,INPUT_SIZE,INPUT_SIZE)
        predicted = get_onehot(predicted.int(),CLASS_NUM,INPUT_SIZE,INPUT_SIZE)
        if b_y.dim() == 3:
            predicted = predicted.unsqueeze(0)
            b_y = b_y.unsqueeze(0)
        dice,dice_class = metrics.multiclass_dice_coeff(predicted,b_y)
        GT.append(cla_true.cpu().numpy())
        Pred.append(y_cla)
        DSC['total_mean'].append(dice)


# seg_gt = torch.cat(seg_gt,dim=0).numpy()
# seg_pred = torch.cat(seg_pred,dim=0).numpy()
# seg_prec,seg_recal = metrics.compute_iou(seg_pred,seg_gt)
dice_arr = DSC['total_mean']
FPS = np.array(FPS)
print('dice:{:.3f},std:{:.3f},miou:{:.3f},std:{:.3f},fps:{:.3f}'.format(np.mean(dice_arr),np.std(dice_arr),np.mean(IOU_s),np.std(IOU_s),FPS.mean()))
y_true = np.array(GT)
y_pred = np.array(Pred)
print(y_pred.shape,y_true.shape)

classify_result = metrics.compute_classify(y_pred,y_true)
print('Acc:{:.3f},Precision:{:.3f},Recall:{:.3f},F1:{:.3f}'.format(classify_result[0],classify_result[1],classify_result[2],classify_result[3]))

# conf_matrix = confusion_matrix(y_true,y_pred)
# print(conf_matrix)
# mean_DSC = np.mean(DSC['total_mean'])
# print('DSC:{:.3f},mIoU:{:.3f},Precision:{:.3f},Recall:{:.3f}'.format(mean_DSC,mean_DSC/(2-mean_DSC),seg_prec,seg_recal))



# print("class0:{:.3f},class1:{:.3f},class2:{:.3f},class3:{:.3f}".format(conf_matrix[0][0]/conf_matrix[0].sum(),\
#     conf_matrix[1][1]/conf_matrix[1].sum(),conf_matrix[2][2]/conf_matrix[2].sum(),conf_matrix[3][3]/conf_matrix[3].sum()))
# plot_matrix(conf_matrix, './fuck.jpg',labels_name=["0", "1", "2", "3"],)
# print(np.mean(mIoU['total_mean']),np.mean(mIoU['0_class']),np.mean(mIoU['1_class']),np.mean(mIoU['2_class']),np.mean(mIoU['3_class']),)
# print(np.mean(IOU))
# print(nums/len(val_loader))
