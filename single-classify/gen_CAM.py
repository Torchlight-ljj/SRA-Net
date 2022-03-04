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

class_nums = 4
NetDict = {
    "resnet50": models.resnet50(pretrained = False, num_classes = 4),
    "densenet121": models.densenet121(pretrained = False, num_classes=4),
    "inception_v3": models.inception_v3(pretrained = False, aux_logits=False, num_classes = 4),
    "mobilenet_v2":models.mobilenet_v2(pretrained = False, num_classes = 4)
}

convs = {
    "resnet50": "layer4",
    "densenet121":"features",
    "inception_v3":"Mixed_7c",
    "mobilenet_v2":"features",
}

net_name = "mobilenet_v2"

def backward_hook(module, grad_in, grad_out):
    grad_block.append(grad_out[0].detach())

def farward_hook(module, input, output):
    fmap_block.append(output)

def cam_show_img(img, feature_map, grads, out_dir):
    H, W, _ = img.shape
    cam = np.zeros(feature_map.shape[1:], dtype=np.float32)		# 4
    grads = grads.reshape([grads.shape[0],-1])					# 5
    weights = np.mean(grads, axis=1)							# 6
    for i, w in enumerate(weights):
        cam += w * feature_map[i, :, :]							# 7
    cam = np.maximum(cam, 0)
    cam = cam / cam.max()
    cam = cv2.resize(cam, (W, H))

    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    cam_img = np.concatenate([img*255,heatmap],0)
    # cam_img = img*255*0.8 + heatmap*0.3
    path_cam_img = out_dir
    cv2.imwrite(path_cam_img, heatmap)

img_list =[
'./data/ori/1/崔永平/崔永平_男_61岁_06716272_检查号_EUS62892-0120211025101959007.jpg',
'./data/ori/2/陈颖/陈颍_女_36岁_06609121_检查号_EUS63598-0120211011150926063.jpg',
'./data/ori/3/杨纪康/杨纪康_男_67岁_QG12027610_检查号_EUS63878-0120211021125432078.jpg',
'./data/ori/3/刘省/刘省_男_53岁_06650440_检查号_EUS63312-0120211021115932033.jpg',
]
count = 0
transform_test = transforms.Compose([
    transforms.ToTensor(),
])
for img_path in img_list:
    
    img_pil = Image.open(img_path)
    ori = transforms.ToTensor()(img_pil)
    ori = transforms.Resize((512,512))(ori)

    original = transform_test(img_pil)
    original = transforms.Resize((512,512))(original)
    img_variable = original.unsqueeze(0).cuda()

    finalconv_name = convs[net_name] 
    net =  NetDict[net_name].cuda()
    net.load_state_dict(torch.load(os.path.join("./",net_name,"model","net_006.pth")))
    net.eval()
    classes = list(key for key in range(class_nums))
    fmap_block = list()
    grad_block = list()
    net._modules.get(finalconv_name).register_forward_hook(farward_hook)
    net._modules.get(finalconv_name).register_backward_hook(backward_hook)

    # forward
    if net_name == "resnet50":
        output,faetures = net(img_variable)
    else:
        output = net(img_variable)
    idx = np.argmax(output.cpu().data.numpy(),axis=1)[0]
    print("predict: {}".format(classes[idx]))

    # backward
    net.zero_grad()
    class_loss = output[0,idx]
    class_loss.backward()

    # generate cam
    grads_val = grad_block[0].cpu().data.numpy().squeeze()
    fmap = fmap_block[0].cpu().data.numpy().squeeze()
    name = img_path.split('/')[-1][:-4]
    # save cam figs
    if not os.path.exists('./grad_cam/'+net_name):
        os.mkdir('./grad_cam/'+net_name)
    cam_show_img(ori.numpy().transpose([1,2,0]), fmap, grads_val, './grad_cam/'+net_name + '/' + finalconv_name+"_"+str(name)+'.jpg')
    count += 1
