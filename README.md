# Autoenocder_Image_similarity_lshashing
    Pytorch based Image similarity retrieval framework trained on a convolutional autoencoder and encoded into 
    hash values using latent similarity hashing for faster inference.

# Dataset
    Provided with a dataset of ~5k 512x512 images of animals
 
 # Training framework
    |
    |------------- model_autoenocer.py : Model class consisting of an autoencoder model with 2 modules encoder and decoder
    |                                    omprising of convolutional, convolutional tranpose, pooling and upsampling operations.
    |
    |------------- dataset_autoencoder.py: Dataset class used to create dataset for training purpose, containes iterarble __getitem__()
    |                                       function and __len__() function
    |
    |------------- early_stop_autoencoder.py: Class contains functions for checking for early stopping condition during training, 
    |                                         checkpointing and saving the model
    |
    |------------- train_autoencoder.py: Contains class that performs training and calls early stopping class
    |
    |
    |-------------- feature_autoencoder.py: Contains class that takes model as input and retrieves enoder features from the trained model.
    |                                       These features have been saved as pickle during th training
    |
    |--------------- lsh_autoencoder.py: imports lshashpy3 class for performing locality sensitive hashing and contains class that finds lsh fetaures
    |                                    of our database and stores the output as pickle 
    |
    |--------------- main_encoder.py : Contains main function that import all the abvove module and performs the training of autoenocder, the outputs 
                                       are saved as checkpoints of model, final saved model, model features saved as pickle and lsh feature saved as pickle44
   
  # Inference Framework
    |
    |
    |---------------- model_autoenocer.py : Model framework class module has been uploaded and saved model weights are loaded into it
    |
    |
    |---------------- feature_aoutoencoder.py : Feature Builder class module has been loaded to find encoder feature representation of input image from
    |                                           the trained model
    |
    |---------------- inferencedata_ae.py : Contains class to rertrieve the image, pre-process the input image and perform health-checks on it.
    |
    |
    |-----------------similar_images.py : Contains class tha performs quering over the lsh databases to retrives and showcase the similar images
    |
    |
    |----------------- inference_autoencoder.ipynb: Contains main module for inference that loads saved model and lsh pickle, call all the above module to perform inferencee
 
 Note : Inference_autoencoder.ipynb has been kept a notebook due to limitation in my current hardware to not support shell. However code for command-line interaface has been commented out and can be used as per requirements
 
 # Auto-Encoder Based Image similarity Retrieval
    Autoencoders are neural networks comprised of both an encoder and a decoder. The goal is to compress your input data 
    with the encoder, then decompress the encoded data with the decoder such that the output is a good/perfect reconstruction 
    of your original input data.
    
    The true worth of the autoencoder lies in the encoder and decoder themselves as separate tools, the encoded representations (embeddings) given by the encoder 
    are magnificent objects for similarity retrieval. Most raw and highly unstructured data is typically embedded in a non-human interpretable vector space representation.
    So instead of operating painstakingly in the space of RGB pixels, we can use a trained encoder to convert the image  to lower-dimensional embedding sitting in a 
    hopefully more meaningful dimensions such as “image brightness”, “head shape”, “location of eyes”, “color of hair”, etc. 
    
    With such a condensed encoding representation, simple vector similarity measures between embeddings (such as cosine similarity) will create much more 
    human-interpretable similarities between images.
    
# Locality Sensitive Hashing
  
    The process of ccalculating distance b/w embeddings of two images is computationally expensive in nature and as a new image embedding have to compare with all the images 
    in the dataset embedding to find the most similar image(nearest neighbor), which in computational complexity notation is an O(DN) problem and will take exponentially
    more time to retrieve similar images as the number of images increases.
    
    To solve this problem, we will use locality sensitive hashing(LSH) which is an approximate nearest neighbor algorithm which reduces the computational complexity to 
    O(log N). LSH generates a hash value for image embeddings while keeping spatiality of data in mind; in particular; data items that are similar in high-dimension 
    will have a higher chance of receiving the same hash value.
    
    Below are the steps on how LSH converts an embedding in a hash of size K-
    > Generate K random hyperplanes in the embedding dimension
    > Check if particular embedding is above or below the hyperplane and assign 1/0
    > Do step 2 for each K hyperplanes to arrive at the hash value
    
    Here, lshashpy3 class has been used to perform lsh

  
