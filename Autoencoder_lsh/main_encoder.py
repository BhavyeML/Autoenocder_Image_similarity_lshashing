# %% [code]
from __future__ import print_function
from __future__ import division
import gc
import time
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split
import dataset_autoencoder
from dataset_autoencoder import Dataset_builder
import model_autoenocer
from model_autoenocer import Net
import train_autoencoder
from train_autoencoder import trainer
import feature_aoutoencoder
from feature_aoutoencoder import feature_builder
import lsh_autoencoder
from lsh_autoencoder import lshasher
import early_stop_autoencoder
from early_stop_autoencoder import EarlyStopping

%matplotlib inline
torch.backends.cudnn.deterministic = True

def main(data_dir,data_path,batch_size,n_epochs):
    
    """ Training Module"""
    
    
    gpu=False
    if torch.cuda.is_available():
        gpu=True
    
    print("Loading Dataset")
    dataset_load=time.time()
    mean=[0.485,0.456,0.406]
    std=[0.5,0.5,0.5]
    composed=transforms.Compose([transforms.Resize(256),transforms.ToTensor(),transforms.Normalize(mean,std)])
    dataset_obj=Dataset_builder(dir_link=data_dir,data_path=data_path,transform=composed)
    dataset= dataset_obj.build_set(on_gpu=gpu)
    train_data,val_data=train_test_split(dataset['set'],test_size=0.1)
    train_data= DataLoader(train_data,batch_size=batch_size)
    val_data= DataLoader(val_data,batch_size=batch_size)
    print("time taken to load dataset {}".format(time.time()-dataset_load))
    del dataset_load
    
    
    print("Loading Model")
    model_time=time.time()
    model= Net(True)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.to(device)
    print("time taken to load model {}".format(time.time()-model_time))
    del model_time
    
    
    print("Instantiating training")
    train_time= time.time()
    train_model=trainer(n_epochs,model)
    model= train_model.train(train_data,val_data)
    print("time taken for training{}".format(time.time()-train_time))
    del train_time
    

    #val=input("If feature representation needs to be retrieved press 'Y'")        
    #print("\n",val)
    
    #if val is'Y'or val is 'y':
    print("Retrieving features")
    feature_time=time.time()
    feature_rep=feature_builder(model)
    features_dict,_=feature_rep.get_feature(dataset['set'],batch_size)
    print("len is {}".format(len(features_dict.items())))
    print("Time taken to retrieve features {}".format(time.time()-feature_time))
    del feature_time
  
    
    print("Hashing features with LSH")
    hash_time=time.time()
    lshash=lshasher(10,5,4096)
    lshash.get_hash(features_dict)

    print("Time taken to retrieve features {}".format(time.time()-hash_time))
    del features_dict
    del hash_time
    gc.collect()
    
    

# %% [code]
if __name__=="__main__":
    main("/kaggle/input","similar-image/dataset/dataset",16,50)