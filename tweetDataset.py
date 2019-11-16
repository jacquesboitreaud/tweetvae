# -*- coding: utf-8 -*-
"""
Created on Sat Oct 26 18:06:44 2019

@author: jacqu

Dataset class and Loader wrapper to pass one-hot tweets to model

"""

import os 
import sys
if __name__ == "__main__":
    sys.path.append("..")
    
import torch
import pandas as pd
import pickle
import numpy as np
import random
import itertools
from collections import Counter

from tweet_converter import *


from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader, Subset



class tweetDataset(Dataset):
    """ 
    pytorch Dataset  
    """
    def __init__(self, 
                data_file="data/rawTwitterData.pickle",
                clean=True,
                debug=False, shuffled=False):
        
        self.path = data_file
        
        # Load and clean tweets
        self.df = pickle.load(open(data_file,'rb'))
        if(clean):
            print('Cleaning data. Will be saved in ./data directory for next time')
            df = cleanTweets(self.df)
            # Save for next time
            pickle.dump(df, open('data/cleanTwitterData.pickle', 'wb'))
        
        self.n = df.shape[0]
        self.max_words = max(df['len']) # Longest tweet in dataset 
        # Get dataset vocabulary 
        self.vocab = self._get_vocab()
        
        print(f'Dataset contains {self.n} tweets, max N_words: {self.max_words}, vocab size : {len(self.vocab)}' )
        
        if(debug):
            # special case for debugging
            pass
            
    def __len__(self):
        return self.n
    
    def __getitem__(self, idx):
        # Returns one-hot tweet tensor , label or only one-hot tweet
        
        tweet, label, length = self.df.iloc[idx]
        
        # Convert tweet to one hot
        onehot = torch.tensor(padded_tweetToVec(tweet, self.vocab, self.max_words), dtype=torch.float)
        
        return onehot, length, torch.tensor(label, dtype=torch.float)
    
    def _get_vocab(self):
        # Returns set of words found in tweets in the dataset 
        sample_vocab = []
        for tweet in self.df['tweet']:
            for word in tweet.split():
                if (word not in sample_vocab):
                    sample_vocab.append(word)
        return sample_vocab
        
class Loader():
    def __init__(self,
                 path='data/rawTwitterData.pickle',
                 batch_size=128,
                 num_workers=1,
                 clean=True,
                 debug=False,
                 shuffled=False):
        """
        Wrapper for test loader, train loader 
        Uncomment to add validation loader 

        """

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.dataset = tweetDataset(path, clean=clean,
                          debug=debug,
                          shuffled=shuffled)

    def get_data(self):
        n = len(self.dataset)
        print(f"Splitting dataset with {n} samples")
        indices = list(range(n))
        # np.random.shuffle(indices)
        np.random.seed(0)
        split_train, split_valid = 0.7, 0.7
        train_index, valid_index = int(split_train * n), int(split_valid * n)
        
        train_indices = indices[:train_index]
        valid_indices = indices[train_index:valid_index]
        test_indices = indices[valid_index:]
        
        train_set = Subset(self.dataset, train_indices)
        #valid_set = Subset(self.dataset, valid_indices)
        test_set = Subset(self.dataset, test_indices)
        print(f"Train set contains {len(train_set)} samples")


        train_loader = DataLoader(dataset=train_set, shuffle=False, batch_size=self.batch_size,
                                  num_workers=self.num_workers)

        # valid_loader = DataLoader(dataset=valid_set, shuffle=True, batch_size=self.batch_size,
        #                           num_workers=self.num_workers, collate_fn=collate_block)
        
        test_loader = DataLoader(dataset=test_set, shuffle=False, batch_size=self.batch_size,
                                 num_workers=self.num_workers)


        # return train_loader, valid_loader, test_loader
        return train_loader, 0, test_loader
    
if(__name__=='__main__'):
    loaders = Loader()
    train, _, test = loaders.get_data()