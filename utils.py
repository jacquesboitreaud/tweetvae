# -*- coding: utf-8 -*-
"""
Created on Thu May  2 10:14:11 2019

@author: jacqu

Utils functions for preprocessing
"""
import numpy as np
from rdkit import Chem
from rdkit.Chem import Draw, QED, Crippen, Descriptors, rdMolDescriptors, GraphDescriptors
import matplotlib.pyplot as plt
import json
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
            
