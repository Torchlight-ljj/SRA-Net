import glob
import random
import os
import sys
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
import json
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import cv2
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

class ListDataset(Dataset):
    def __init__(self, list_path,transform = None, image_size = 256,train=True, is_ours = False):
        self.labels = []
        self.batch_count = 0
        self.transform = transform
        self.train = train
        self.image_size = image_size
        self.is_ours = is_ours
#         self.label_json = json.load(open('./four_classes.json','r'))
        with open(list_path, "r") as file:
            self.img_files = file.readlines()
        # if self.train == True:
        #     for path in self.img_files:
        #         path = path.rstrip()
        #         name = path.split('/')[2]
        #         name = name.split('.')[0]
        #         cla = int(self.label_json[name])
        #         self.labels.append(cla)
        # self.cache = {}

    # def __getitem__(self, index):
    #     if index not in self.cache:
    #         self.cache[index] = self._getitem__(index)
    #     return self.cache[index]


    def __getitem__(self, index):
        # ---------
        #  Label
        # ---------
        temp_path = self.img_files[index % len(self.img_files)].rstrip()
        temp_path = '../..'+temp_path[1:]
        label_path = temp_path.replace("jpg","png").replace("ori","mask")
        classes = int(label_path.split('/')[4])
        label_size = None
        if classes == 0:
            label = torch.zeros((self.image_size,self.image_size))
            label_size = label.shape
            label = torch.Tensor(label).unsqueeze(0)
        else:
            if self.is_ours:
                label = (cv2.imread(label_path,0)/255)*1
            else:
                label = (cv2.imread(label_path,0)/255)*classes
            label_size = label.shape
            label = torch.Tensor(label).unsqueeze(0)
            # if self.train:
            #     label = self.transform(label)
            # else:
            label = transforms.Resize((self.image_size,self.image_size))(label)

        # ---------
        #  Image
        # ---------
        img_path = self.img_files[index % len(self.img_files)].rstrip()
        img_path = '../../'+img_path[1:]
        # Extract image as PyTorch tensor
        img = Image.open(img_path)
        if self.transform is not None:
            img = transforms.ToTensor()(img)
            img = transforms.Resize((label_size[0],label_size[1]))(img)
            img = self.transform(img)
            # img,_ = pad_to_square(img,0)
            # img = resize(img,(self.image_size,self.image_size))
        else:
            # img = transforms.ToTensor()(img)[0].unsqueeze(0)
            img = transforms.ToTensor()(img)
            img = transforms.Resize((label_size[0],label_size[1]))(img)
            img = transforms.Resize((self.image_size,self.image_size))(img)
            # img = resize(img,(self.image_size,self.image_size) )
        # except Exception as e:
        #     print(temp_path)
        #     pass
        return img, label, classes
    # def collate_fn(self, batch):
    #     if self.train == True:
    #         paths, imgs, targets = list(zip(*batch))
    #         imgs = torch.stack([img for img in imgs])
    #         targets = [target for target in targets]
    #         targets = torch.cat(targets,0)
    #         return paths, imgs, targets
    #     else:
    #         paths, imgs = list(zip(*batch))
    #         imgs = torch.stack([img for img in imgs])
    #         return paths, imgs           

    def __len__(self):
        return len(self.img_files)
