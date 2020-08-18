
from io import BytesIO
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import re
import requests
import torch
from torch.utils.data import Dataset
from torchvision import transforms

class get_file(Dataset):
    """ Class encapsulating function to get inference image using url or local directory path & perform transformations and data checks"""
    
    def __init__(self,input_size=512,transform=None):
        self.in_size=input_size
        self.transform=transform
        self.device=torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.dload=False
        self.check=False
    
    def __getitem__(self, file):
        
        img_set=[]
        
        if re.search('^Http',file):
            self.dload=True
        
        try:
            print("Opening File")
            if self.dload is True:
                response = requests.get(file,stream=True)
                self.img = Image.open(BytesIO(response.content))
                plt.imshow(self.img)
                
            else:
                self.img=Image.open(file)
                plt.imshow(self.img)
                
        except Exception as e:
            print(e)
                 
        self.check=self.check_size()
        if self.check is not True: 
             raise IOError(f"Incorrect dimensions. Insert an image of dimension 512x512")
        
        
        if self.transform:
            self.img=self.transform( self.img)
            self.img.to(self.device)
        img_set.append((file,self.img))
                
        return  img_set
    
    def check_size(self):
        if self.img.size == (self.in_size,self.in_size):
            return True