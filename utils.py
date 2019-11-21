# -*- coding: utf-8 -*-
"""
Created on Thu May  2 10:14:11 2019

@author: jacqu

variable wrapper for pytorch + Beam Search utils for decoding with beam search
"""
import numpy as np
import pandas as pd
import torch

from torch.utils.data import Dataset

def Variable(tensor):
    """Wrapper for torch.autograd.Variable that also accepts
       numpy arrays directly and automatically assigns it to
       the GPU. Be aware in case some operations are better
       left to the CPU."""
    if isinstance(tensor, np.ndarray):
        tensor = torch.from_numpy(tensor)
    if (torch.cuda.is_available()) :
        return torch.autograd.Variable(tensor).cuda()
    else:
        return torch.autograd.Variable(tensor)
    

class BeamSearchNode():
    def __init__(self, h, rnn_in, score, sequence):
        self.h=h
        self.rnn_in=rnn_in
        self.score=score
        self.sequence=sequence
        self.max_len = 60 

    def __lt__(self, other): # For x < y
        # To break score equality cases randomly
        return True
            
def prepare_set():
    # Prepares training set when just downloaded from web
    df=pd.read_csv('data/full_train_set.csv', header=None, encoding="ISO-8859-1")
    df.columns = ['label', 'id', 'date', 'query', 'auth_name', 'tweet' ]
    df.to_csv('data/full_train_set.csv')

if(__name__=='__main__'):
    df=prepare_set()