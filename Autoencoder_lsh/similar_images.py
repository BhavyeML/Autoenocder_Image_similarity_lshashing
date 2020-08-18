get_ipython().system('pip install lshashpy3')
from lshashpy3 import LSHash
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

class similar_item(object):
    
    """ Class encapsulating functions related to retrieving images and showing them through matplotlib"""
    
    def __init__(self, feature,lsh):
        super(similar_item,self).__init__()
        self.feature=feature
        self.lsh_variable=lsh
    
    def __getitem__(self,n_items=5):
        

        response=self.lsh_variable.query(self.feature[list(self.feature.keys())[0]].flatten(),num_results=n_items,distance_func='hamming')
        self.__showitem__(response,n_items)
    
    def __showitem__(self,response,n_items):
        
        columns = 3
        rows = int(np.ceil(n_items+1/columns))
        fig=plt.figure(figsize=(2*rows, 3*rows))
        for i in range(1, columns*rows +1):
            if i<n_items+1:
                img = Image.open(response[i-1][0][1])
                fig.add_subplot(rows, columns, i)
                plt.imshow(img)
        return plt.show()