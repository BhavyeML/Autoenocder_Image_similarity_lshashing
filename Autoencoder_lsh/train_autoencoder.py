from __future__ import print_function
from __future__ import division
import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import early_stop_autoencoder
from early_stop_autoencoder import EarlyStopping

class trainer():
    
    """ Training class"""
    def __init__(self,n_epochs,model):
        
        self.epochs=n_epochs
        
        self.model=model
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.model.to(self.device)
        
        
    
    def train(self,train_data,val_data):
        
        early_stopping = EarlyStopping( self.epochs, patience=14, verbose=True)
        loss_list=[]
        loss_vallist=[]
        self.model.train()
        criterion=nn.MSELoss()
        optimizer=torch.optim.Adam(self.model.parameters(),lr=0.001)
        
        for epoch in range(self.epochs):
            
            print('Epoch {}/{}'.format(epoch, self.epochs))
            loss_sublst=[]
            loss_valsublst=[]
            
            for x,y in train_data:
                y=y.to(self.device)
                self.model.train()
                optimizer.zero_grad()
                z=self.model(y)
                loss=criterion(z,y)
                loss_sublst.append(loss.data.item())
                loss.backward()
                optimizer.step()
        
            print('train Loss: {}'.format(np.mean(loss_sublst)))
            loss_list.append(np.mean(loss_sublst))
            
            with torch.no_grad():
                for x,y in val_data:
                    y=y.to(self.device)
                    self.model.eval()
                    z=self.model(y)
                    loss_val = criterion(z, y)
                    loss_valsublst.append(loss_val.data.item())

            print('val Loss: {}'.format(np.mean(loss_valsublst)))
            loss_vallist.append(np.mean(loss_sublst))
            early_stopping(np.mean(loss_valsublst),self.model,optimizer,epoch)
        
            if early_stopping.early_stop:
                print("Early stopping")
                break
                
        self.plot_loss(loss_list,phase='train')
        self.plot_loss(loss_vallist,phase='val')
        return self.model
    
    def plot_loss(self,loss_list,phase):
        plt.plot(loss_list,label='loss')
        plt.title('loss per epoch')
        plt.xlabel('epoch')
        plt.ylabel(phase+'loss')
        plt.legend()
        plt.show()