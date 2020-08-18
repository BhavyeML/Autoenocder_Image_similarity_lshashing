
get_ipython().system('pip install lshashpy3')
import pickle
from lshashpy3 import LSHash
from tqdm import tqdm_notebook
import torch

class lshasher():
    """ Class perforing latent similarity hashing and saving the output of dataset as pickle """
    def __init__(self,K,L,dim):
        self.K=K
        self.L=L
        self.dim=dim
    
    def get_hash(self,embed_dic):
        
        lsh=LSHash(hash_size=self.K,input_dim=self.dim,num_hashtables=self.L)
        for img_path, vec in tqdm_notebook(embed_dic.items()):
            lsh.index(vec.flatten(), extra_data=img_path)
        self.save_hash(lsh)
    
    def save_hash(self,lsh):
        path="lsh.p"
        print("Saving features as a pickle {}".format(path) )
        pickle.dump(lsh, open(path, "wb"))