import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter   
import argparse
from torch.autograd import Variable
import numpy as np
import os
from dataset import *
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from PIL import Image
from dataset import resize
import torchvision.models as models
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import metrics
EPOCH = 10
pre_epoch = 0  
BATCH_SIZE = 1      
LR = 0.001  

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
setup_seed(20)


class Noise(object):
    def __call__(self, tensor):
        return tensor+torch.FloatTensor(np.random.normal(0,0.05,tensor.shape))

transform_train = transforms.Compose([
    transforms.RandomRotation((0,60)),
    # transforms.RandomResizedCrop(256,scale=(0.6,1.0)),
    transforms.RandomAffine(5,),
    transforms.RandomHorizontalFlip(), 
    transforms.ToTensor(),
    transforms.Normalize((0.1642,), (0.317,)), 
    # Noise(),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1642,), (0.317,)), 
])

trainset = ListDataset('./train.txt',transform=None, image_size=512, train=True)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=8, shuffle=False, num_workers=10)  

testset = ListDataset('./test.txt',transform=None,image_size=512, train=True)
testloader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False, num_workers=10)

NetDict = {
   "resnet50": models.resnet50(pretrained = False, num_classes = 4),
   "squeezenet1_1": models.squeezenet1_1(pretrained = False, num_classes = 4)
    "densenet121": models.densenet121(pretrained = False, num_classes=4),
    "inception_v3": models.inception_v3(pretrained = False, aux_logits=False, num_classes = 4),
    "mobilenet_v2":models.mobilenet_v2(pretrained = False, num_classes = 4)
}

if __name__ == "__main__":
    best_acc = 90 
    first_acc = 0
    acc = 0
    for net_name in NetDict.keys():
        if not os.path.exists(net_name):
            os.mkdir(net_name)
        
        net = NetDict[net_name].cuda()
        net.load_state_dict(torch.load(os.path.join('./pretrained',net_name+'.pth')), False)
        criterion = nn.CrossEntropyLoss()
        feature_loss = MultiLoss()
        optimizer = optim.SGD([{"params": net.parameters()},{"params": feature_loss.parameters()}], lr=0.001, momentum=0.9, weight_decay=5e-4) 
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
        writer = SummaryWriter(net_name+'/logs')
        print("Start Training!")
        step = 0
        
        with open(net_name+"/acc.txt", "w") as f:
            for epoch in range(pre_epoch, EPOCH):
                print(net_name+'\nEpoch: %d' % (epoch + 1))
                net.train()
                correct = 0
                total = 0
                correct = float(correct)
                total = float(total)
                for i, (_, inputs, labels) in enumerate(trainloader):
                    batch = len(inputs)
                    inputs = inputs.cuda()
                    labels = labels.cuda() 
                    optimizer.zero_grad()
                    # forward + backward
                    outputs = net(inputs)
                    loss = criterion(outputs, labels)

                    writer.add_scalar('loss', loss, step)
                    writer.add_scalar('lr',optimizer.state_dict()['param_groups'][0]['lr'],step)
                    loss.backward()
                    optimizer.step()
                    step += 1
                    print('batch:%d/%d, loss:%.4f,epoch:%d.'%(i,len(trainloader),loss, epoch))
                scheduler.step()
                trues = []
                preds = []
                if True:
                    print("Waiting Test!")
                    with torch.no_grad():

                        for i, (_, images, labels)  in enumerate(testloader):
                            net.eval()
                            images, labels = images.cuda(), labels.cuda()
                            outputs = net(images)
                            predicted = torch.argmax(outputs, 1)
                            trues.append(labels)
                            preds.append(predicted)

                        trues = torch.cat(trues,dim=0).cpu().numpy()
                        preds = torch.cat(preds,dim=0).cpu().numpy()
                        classify_result = metrics.compute_classify(preds,trues)
                        print('EPOCH:{:5d},Acc:{:.3f},Precision:{:.3f},Recall:{:.3f},F1:{:.3f}.\n'.format(epoch + 1,classify_result[0],classify_result[1],classify_result[2],classify_result[3]))
                        if epoch % 5 == 0:
                            print('Saving model......')
                            if not os.path.exists(net_name+'/model'):
                                os.mkdir(net_name+'/model')
                            torch.save(net.state_dict(), '%s/net_%03d.pth' % (net_name+'/model', epoch + 1))
                        f.write('EPOCH:{:5d},Acc:{:.3f},Precision:{:.3f},Recall:{:.3f},F1:{:.3f}.\n'.format(epoch + 1,classify_result[0],classify_result[1],classify_result[2],classify_result[3]))
                        f.flush()
            
    print("Training Finished!!!")