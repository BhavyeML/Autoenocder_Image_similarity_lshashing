from __future__ import print_function
from __future__ import division
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pickle
from tqdm import tqdm_notebook

class feature_builder(object):
    
    """ Class encapsulating function related to retrieving feature vectors of images for training and inference and saving th feature pickle during training"""
    def __init__(self,model):
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.model=model.to(self.device)
        
    def get_feature(self,data,batch_size,train=True):
    
        features=[]
        feat_data=DataLoader(data,batch_size=batch_size)
        img_path = [str(x[i]) for  x,y in list(feat_data) for i in range(len(x))]
        
        for file,x in feat_data:
            x=x.to(self.device)
            self.model.eval()
            z=self.model.encoder(x)
            z=z.detach().cpu().numpy()
            for i in range(z.shape[0]):
                features.append(z[i])
                
        feature_dict = dict(zip(img_path,features))      
        if train is True:
            self.save_feature(feature_dict)
        return feature_dict,features
    
    def save_feature(self,feature_dict):
        
        path="feature_dict.p"
        print("Saving features as a pickle {}".format(path) )
        pickle.dump(feature_dict, open(path, "wb"))