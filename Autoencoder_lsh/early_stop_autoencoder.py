# %% [code]
from __future__ import print_function
from __future__ import division
import os
import gc
import shutil
import time
import numpy as np
import matplotlib.pyplot as plt
import PIL
import torch
import torch.nn as nn
from PIL import Image
import torch.nn.functional as f
from torch.utils.data import DataLoader
from torchvision import transforms
import glob
import pickle
from tqdm import tqdm_notebook
from sklearn.model_selection import train_test_split


torch.backends.cudnn.deterministic = True


class EarlyStopping():
    """Early stops the training if validation loss doesn't improve after a given patience. Also used to save checkpoint anf final model"""
    def __init__(self,n_epochs, patience=7, verbose=False):
        """
        Args:
            patience (int): How long to wait after last time validation acc improved.
                            Default: 4
            verbose (bool): If True, prints a message for each validation acc improvement. 
                            Default: False
        """
        print('Early Stopping Activated')
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = np.inf
        self.early_stop = False
        self.epochs=n_epochs

        
    def __call__(self, val_loss, model,optimizer,epoch):
        
        print('checking early stop')
        score1 =  val_loss
        
        if epoch == self.epochs-1:
            self.save_model(score1, model,optimizer,epoch)
            
        else:
            if (score1+0.0009)<self.best_score:
                self.counter += 1
                if self.counter >= self.patience:
                    print("Desired Loss achieved at epoch {}".format(epoch))
                    self.early_stop = True 
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
                self.save_checkpoint(score1, model,optimizer,epoch)   

            else:
                  self.counter=0  
    
    def save_checkpoint(self, val_loss, model,optimizer,epoch):
        
        if self.verbose:
            print(f'Validation loss decreased ({self.best_score:.6f} --> {val_loss:.6f}).  Saving model ...')
        
        if os.path.isdir("checkpoints"):
            pass
        else:
            shutil.os.mkdir("checkpoints")
        
        if self.early_stop is True:
            path='Autoencoder.pt'
        else:
            path ='checkpoints/checkpt_epoch_{}.pt'.format(epoch)
            
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': val_loss,
            }, path)
        
        self.best_score=val_loss
    
    def save_model(self,val_loss, model,optimizer,epoch):

        path="Autoencoder.pt"
        print("Saving Final Model")
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': val_loss,
            }, path)