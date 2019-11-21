# -*- coding: utf-8 -*-
"""
Created on Thu Nov 21 10:29:55 2019

@author: jacqu

Evaluate model + experimental ideas
    
    0/ Reconstruction performance on independent test set 
    1/ Latent space structure : Select 1000 tweets of each topic and plot in latent space (PCA)
    2/ Sample randomly in latent space and decode 
    3/ Write dumb / noisy tweets and encode in latent space / look at where they are encoded
    4/ Latent representation used as features for a simple classification model 
    
!! All words in tweets should be in the training set vocabulary, otherwise it won't work. 
Training set vocabulary is loaded from .npy file. 
    
"""

import pickle

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.utils.data
from torch import nn

from model_birnn import tweetVAE, Loss, set_kappa, load_my_state_dict, lr_decay
from tweetDataset import tweetDataset, Loader
from utils import *


# Loading the pretrained model and vocabulary
print("loading vocabulary")
vocab = pickle.load(open("./saved_model_w/vocabulary.pickle","wb"))

voc_size = len(vocab.keys())
max_words_in_tweet = 50

print("loading trained model")
model_params={'MAX_LEN': 52, # + start-of-sentence token and end-of-sentence token
              'vocab_size': voc_size,
              'device': 'cuda' if torch.cuda.is_available() else 'cpu',
              'N_properties':1,
              'N_topics':5} 
model = tweetVAE(**model_params ).to(model_params['device']).float()
load_my_state_dict(model, torch.load('./saved_model_w/first_try.pth' , map_location=map))

# Latent space structure : 
