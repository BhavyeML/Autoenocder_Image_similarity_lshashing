from __future__ import print_function
from __future__ import division
import torch
import torch.nn as nn
import torch.nn.functional as f


class Net(nn.Module):
    """ Pytoch based Autoencoder model class used for training on the dataset"""
    def __init__(self,feature_extracting=True):
        super(Net,self).__init__()
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.feature_extracting=feature_extracting
        
    
        self.encoder = nn.Sequential(nn.Conv2d(in_channels=3,out_channels=16,kernel_size=(3,3),stride=1,padding=1),  #Input Size: (32,3,256,256)
                                    nn.ReLU(True),
                                    nn.MaxPool2d(2, stride=2),  
                                    nn.Conv2d(in_channels=16,out_channels=8,kernel_size=(3,3), stride=1,padding=1),  
                                    nn.ReLU(True),
                                    nn.MaxPool2d(2, stride=2),
                                    nn.Conv2d(in_channels=8,out_channels=4,kernel_size=(3,3), stride=1,padding=1),  
                                    nn.ReLU(True),
                                    nn.MaxPool2d(2, stride=2)  #Output Size: (32,4,32,32)
                                    )
        self.encoder=self.encoder.to(self.device)
        self.decoder = nn.Sequential(nn.ConvTranspose2d(in_channels = 4,out_channels=8,kernel_size=(3,3),stride=1,padding=1),  #Input Size (32,4,32,32)
                                     nn.ReLU(True),
                                     nn.Upsample(scale_factor=2, mode='bicubic'),
                                     nn.ConvTranspose2d(in_channels=8,out_channels=16,kernel_size=(3,3),stride=1,padding=1),  
                                     nn.ReLU(True),
                                     nn.Upsample(scale_factor=2, mode='bicubic'), 
                                     nn.ConvTranspose2d(in_channels=16,out_channels=8,kernel_size=(3,3), stride=1,padding=1), 
                                     nn.ReLU(True),
                                     nn.Upsample(scale_factor=2, mode='bicubic'),#(N,8,512,512)
                                     nn.ConvTranspose2d(in_channels=8,out_channels=3,kernel_size=(3,3), stride=1,padding=1),  #Output Size(32,3,256,256)
                                     nn.Tanh()
                                     )
        
        self.decoder=self.decoder.to(self.device)
        
    
    def forward(self,x):
        x=x.to(self.device)
        x=self.encoder(x)
        x=self.decoder(x)
        return x