from __future__ import print_function
from __future__ import division
import os
import numpy as np
import PIL
import torch
import torch.nn as nn
from PIL import Image
import torch.nn.functional as f
from torch.utils.data import Dataset
from torchvision import transforms
import glob


class Dataset_builder(Dataset):
    """ Dataset building class with iterative properties and built on top of built-in Pytorch Dataset class"""
    
    def __init__(self,dir_link,data_path,transform=None):
        
        
        self.transform=transform
        self.data_dir=dir_link
        self.data_path=data_path
        self.dataset=os.path.join(self.data_dir,self.data_path)
        if not os.path.isdir(self.dataset):
            raise IOError(f"{self.dataset} doesn't exist")
        self.len=len(glob.glob(os.path.join(self.dataset,"*.jpg")))
    
    def __len__(self):
        
        return self.len
    
    def __getitem__(self,idx,on_gpu):
        
        image_name=os.path.join(self.dataset,str(idx)+".jpg")
        
        image=Image.open(image_name)
        
        if self.transform:
            image=self.transform(image)
            if on_gpu:
                image.to(torch.device('cuda'))
        return image,image_name
    
    def build_set(self,on_gpu=False):
        
        img_set={"set":[]}
        for i in range(0,self.len):
            image,image_name=self.__getitem__(i,on_gpu)
            img_set["set"].append((image_name,image))
        return img_set
    