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
                data_file,
                clean=True,
                debug=False, shuffled=False, n_tweets=200000):
        
        self.path = data_file
        
        # Load and clean tweets
        self.df = pd.read_csv(data_file, nrows=n_tweets)
        if(clean):
            print('Cleaning data. Will be saved in ./data directory for next time')
            df = cleanTweets(self.df) # filter out weird symbols 
            df = clean_dataframe(df) # remove nan values 
            # Save for next time
            df.to_csv(data_file)
            self.df=df
        

        self.n = self.df.shape[0]
        self.max_words = max(self.df['len']) # Longest tweet in dataset 
        # Get dataset vocabulary 
        vocab = self._get_vocab()
        self.voc_len = len(vocab)
        self.words_to_ids = {w:i for (i,w) in enumerate(vocab)}
        self.ids_to_words = {i:w for (i,w) in enumerate(vocab)}
        # EOS token 
        self.EOS_token = len(vocab) 
        self.words_to_ids['EOS']=self.EOS_token
        self.ids_to_words[self.EOS_token]='EOS'
        
        print(f'Dataset contains {self.n} tweets, max N_words: {self.max_words}, vocab size : {self.voc_len}' )
        
        if(debug):
            # special case for debugging
            pass
            
    def __len__(self):
        return self.n
    
    def __getitem__(self, idx):
        # Returns vector of word-indices and label
        
        tweet, label, length = self.df.loc[idx,['tweet','label','len']]
        # Convert tweet to one hot
        word_ids = torch.tensor([self.words_to_ids[w] for w in tweet], dtype=torch.long)
        word_ids.append(self.EOS_token)
        
        return word_ids, torch.tensor(length, dtype=torch.long), torch.tensor(label, dtype=torch.long)
    
    def _get_vocab(self):
        # Returns set of words found in tweets in the dataset 
        sample_vocab = []
        for tweet in self.df['tweet']:
            if(type(tweet)!=str):
                print('type error, ', type(tweet), tweet)
            for word in tweet.split():
                if (word not in sample_vocab):
                    sample_vocab.append(word)
        return sample_vocab
        
class Loader():
    def __init__(self,
                 path='data/train.csv',
                 batch_size=128,
                 num_workers=1,
                 clean=True,
                 debug=False,
                 shuffled=False,
                 max_n = 200000):
        """
        Wrapper for test loader, train loader 
        Uncomment to add validation loader 

        """

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.dataset = tweetDataset(path, clean=clean,
                          debug=debug,
                          shuffled=shuffled,
                          n_tweets=max_n)

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