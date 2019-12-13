# -*- coding: utf-8 -*-
"""
Created on Wed Nov  6 18:44:04 2019
@author: jacqu
Trains sequence-to-sequence VAE to learn tweet embedding  
"""
import sys
import pickle
import numpy as np
import torch
import torch.utils.data
from torch import nn, optim
import torch.nn.utils.clip_grad as clip
import torch.nn.functional as F

from model_birnn import tweetVAE, Loss, set_kappa, load_my_state_dict, lr_decay
from tweetDataset import tweetDataset, Loader
from utils import *
import nltk

from matplotlib import pyplot as plt
from sklearn.decomposition import PCA 



if(__name__=="__main__"):
# config
    n_epochs = 200 # epochs to train
    batch_size = 64
    # File to save the model's weights
    SAVE_FILENAME='./saved_model_w/first_try.pth'
    LOGS='../data/logs_first_try.npy'
    # To load model 
    SAVED_MODEL_PATH ='./first_model/model_our_dataset.pth'
    # SAVED_MODEL_PATH ='./big_model/model.pth'

    LOAD_MODEL=True # set to true to load pretrained model
    SAVE_MODEL=False 
    clean_data=False # Set to true if first time the dataset is processed
    remove_unfrequent=False
    

    #Load train set and test set
    loaders = Loader(path = 'data_clean.csv',
                     num_workers=0, 
                     batch_size=batch_size, 
                     clean= clean_data,
                     remove=remove_unfrequent,
                     max_n=1000000, # max nbr of datapoints
                     debug=False) # test set = train set (overfitting on purpous)
    
    train_loader, _, test_loader = loaders.get_data()
    # Save vocabulary for later (evaluation):
    # pickle.dump(loaders.dataset.words_to_ids,open("./first_model/vocabulary_our_d.pickle","wb"))
    # glove_matrix = loaders.get_glove_matrix('data/glove')
    # pickle.dump(glove_matrix,open("./first_model/glove_matrix_our_d.pickle","wb"))
    # glove_matrix = pickle.load( open("./first_model/glove_matrix_our_d.pickle", "rb" ))

    words_to_ids = pickle.load(open("./first_model/vocabulary_our_d.pickle","rb"))
    glove_matrix = pickle.load(open("./first_model/glove_matrix_our_d.pickle","rb"))

    # words_to_ids = pickle.load(open("./big_model/vocabulary.pickle","rb"))
    # glove_matrix = pickle.load(open("./big_model/glove_matrix.pickle","rb"))


    

    ids_to_words = {v: k for k, v in words_to_ids.items()}
    voc_size = len(words_to_ids.keys())
    #...

    # ADD THIS : 
    loaders.dataset.words_to_ids=words_to_ids
    loaders.dataset.ids_to_words=ids_to_words
    loaders.dataset.voc_size = len(words_to_ids.keys())
    
    #Model & hparams
    model_params={'MAX_LEN': loaders.dataset.max_words,
                  'vocab_size': loaders.dataset.voc_len,
                  'device': 'cuda' if torch.cuda.is_available() else 'cpu',
                  'N_properties':1,
                  'N_topics':5,
                  'weights_matrix': glove_matrix} 
    
    model = tweetVAE(**model_params ).to(model_params['device']).float()
    
    if(LOAD_MODEL):
        print("loading network coeffs")
        model.load_state_dict(torch.load(SAVED_MODEL_PATH,map_location = 'cpu'))
    
    import_latent = pickle.load(open('our_latent.pickle',"rb"))
    target = pickle.load(open('our_y.pickle',"rb"))

    pca = PCA(n_components=2)
    principalComponents = pca.fit_transform(import_latent)
    word_idces = []
    with torch.no_grad():
        for batch_idx, data in enumerate(train_loader):      
            print("Current batch #: ", batch_idx)
            word_idces.append(data[0].to(model_params['device'])) 
    print(len(word_idces))
            

    # output = model.decode(import_latent[35326])
    # print(output)


# index = 0
# for row in principalComponents: 
#     if row[0] > 7 and row[0] < 8: 
#         if row[1] > 1.2 and row[1] < 1.4: 
#             print(index)
#     index+=1


# target_names = ['Politics', 'Sports', 'Movies', 'Companies', 'General']
# target_np = np.asarray(target) 
# target_final = []
# for i in target_np: 
#     target_final.append(target_names[int(i)])

# pca = PCA(n_components=2)
# principalComponents = pca.fit_transform(import_latent)

# print(pca.explained_variance_)


# principalDf = pd.DataFrame(data = principalComponents, columns = ['principal component 1', 'principal component 2'])
# targetDf = pd.DataFrame(data = np.asarray(target_final), columns = ['target'])
# finalDf = pd.concat([principalDf, targetDf], axis = 1)

# fig = plt.figure(figsize = (8,8))
# ax = fig.add_subplot(1,1,1) 
# ax.set_xlabel('Principal Component 1', fontsize = 15)
# ax.set_ylabel('Principal Component 2', fontsize = 15)
# ax.set_title('2 component PCA on Latent Space', fontsize = 20)

# axes = plt.gca()
# axes.set_xlim([-10,17])
# axes.set_ylim([-3,4])

# targets = target_names
# colors = ['r', 'g', 'b', 'y', 'c']
# for target, color in zip(targets,colors):
#     indicesToKeep = finalDf['target'] == target
#     ax.scatter(finalDf.loc[indicesToKeep, 'principal component 1']
#                 , finalDf.loc[indicesToKeep, 'principal component 2']
#                 , c = color
#                 , s = 50)

# ax.legend(targets)

# # Turn on the minor TICKS, which are required for the minor GRID
# ax.minorticks_on()

# # Customize the major grid
# ax.grid(which='major', linestyle='-', linewidth='0.5', color='black')
# # Customize the minor grid
# ax.grid(which='minor', linestyle=':', linewidth='0.5', color='black')

# plt.savefig('pca_import.png')   
# plt.show()

