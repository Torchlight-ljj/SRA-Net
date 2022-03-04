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
    def __init__(self, list_path,transform = None, image_size = 512,train=True):
        self.labels = []
        self.batch_count = 0
        self.transform = transform
        self.train = train
        self.image_size = image_size
        with open(list_path, "r") as file:
            self.img_files = file.readlines()
            # if self.train == True:
            #     random.shuffle(self.img_files)
        if self.train == True:
            for path in self.img_files:
                  path = path.rstrip()
                  name = path.split('/')[3]
                  cla = int(name)
                  self.labels.append(cla)
        self.cache = {}

    # def __getitem__(self, index):
    #     if index not in self.cache:
    #         self.cache[index] = self._getitem__(index)
    #     return self.cache[index]


    def __getitem__(self, index):
        # ---------
        #  Image
        # ---------
        img_path = self.img_files[index % len(self.img_files)].rstrip()
        # Extract image as PyTorch tensor
        img = Image.open(img_path)
        if self.transform is not None:
            img = self.transform(img)
        else:
            img = transforms.ToTensor()(img)
            img = transforms.Resize((self.image_size,self.image_size))(img)


        # ---------
        #  Label
        # ---------
        if self.train == True:
            targets = self.labels[index]
            return img_path, img, targets
        else:
            return img_path, img
        
    def __len__(self):
        return len(self.img_files)
    
