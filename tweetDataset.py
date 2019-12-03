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
from torch.nn.utils.rnn import pad_sequence

import pandas as pd
import numpy as np

from tweet_converter import *


from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader, Subset

import pickle 
import bcolz



class tweetDataset(Dataset):
    """ 
    pytorch Dataset  
    """
    def __init__(self, 
                data_file,
                clean=True,
                remove=True,
                debug=False, 
                shuffled=False, 
                n_tweets=200000):
        
        self.path = data_file
        
        # Load and clean tweets
        self.df = pd.read_csv(data_file, nrows=n_tweets)
        if(clean):
            print('Cleaning data. Will be saved in ./data directory for next time')
            self.df = clean_dataframe(self.df) # remove nan values 
            self.df = cleanTweets(self.df) # filter out weird symbols 
            if(remove):
                self.df = removeLowFreqWords(self.df,10) # Minimum count = 10 
            # Save for next time
            self.df.to_csv(data_file)
        

        self.n = self.df.shape[0]
        self.max_words = max(self.df['len']) # Longest tweet in dataset 
        # Get dataset vocabulary 
        self.vocab = self._get_vocab()
        self.words_to_ids = {w:i+3 for (i,w) in enumerate(self.vocab)}
        self.ids_to_words = {i+3:w for (i,w) in enumerate(self.vocab)}
        self.words_to_ids['<pad>']=0
        self.ids_to_words[0]='<pad>'
        self.words_to_ids['<eos>']=1
        self.ids_to_words[1]='<eos>'
        self.words_to_ids['<start>']=2
        self.ids_to_words[2]='<start>'
        
        # Number of tokens in vocabulary
        self.voc_len = len(self.vocab) + 3
        print(f'Dataset contains {self.n} tweets, max N_words: {self.max_words}, vocab size : {self.voc_len}' )
        
        
        if(debug):
            # special case for debugging
            pass
            
    def __len__(self):
        return self.n
    
    def __getitem__(self, idx):
        # Returns vector of word-indices and label
        
        tweet, label, length = self.df.loc[idx,['tweet','label','len']]
        word_ids=[self.words_to_ids[w] for w in tweet.split()]
        word_ids.append(self.words_to_ids['<eos>']) # append EOS
        length=len(word_ids) # account for EOS token
        
        word_tensor = torch.zeros(self.max_words+1, dtype=torch.long) # words + EOS token
        word_tensor[:length]=torch.tensor(word_ids,dtype=torch.long)
        
        return word_tensor, torch.tensor(length, dtype=torch.long), torch.tensor(label, dtype=torch.long)
    
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
                 path,
                 #path="test.csv",
                 batch_size=128,
                 num_workers=0,
                 clean=True,
                 remove=True,
                 debug=False,
                 shuffled=True,
                 max_n = 200000):
        """
        Wrapper for test loader, train loader 
        Uncomment to add validation loader 

        """

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.dataset = tweetDataset(path, clean=clean,
                                    remove=remove,
                          debug=debug,
                          shuffled=shuffled,
                          n_tweets=max_n)
        self.embedding_dim = 50
        self.debug=debug

    def get_data(self):
        n = len(self.dataset)
        print(f"Splitting dataset with {n} samples")
        indices = list(range(n))
        # np.random.shuffle(indices)
        np.random.seed(0)
        split_train, split_valid = 0.95, 0.95
        train_index, valid_index = int(split_train * n), int(split_valid * n)
        
        train_indices = indices[:train_index]
        valid_indices = indices[train_index:valid_index]
        test_indices = indices[valid_index:]
        
        train_set = Subset(self.dataset, train_indices)
        #valid_set = Subset(self.dataset, valid_indices)
        if(not self.debug):
            test_set = Subset(self.dataset, test_indices)
        else:
            test_set = Subset(self.dataset,train_indices)
        print(f"Train set contains {len(train_set)} samples")


        train_loader = DataLoader(dataset=train_set, shuffle=False, batch_size=self.batch_size,
                                  num_workers=self.num_workers,drop_last=True)

        # valid_loader = DataLoader(dataset=valid_set, shuffle=True, batch_size=self.batch_size,
        #                           num_workers=self.num_workers, collate_fn=collate_block)
        test_loader = DataLoader(dataset=test_set, shuffle=False, batch_size=self.batch_size,
                                 num_workers=self.num_workers,drop_last=True)


        # return train_loader, valid_loader, test_loader
        return train_loader, 0, test_loader
    
    def get_glove_matrix(self,glove_dir):
        # Returns embeddings weights matrix for vocabulary.
        vectors = bcolz.open(f'{glove_dir}/27B.50.dat')[:]
        words = pickle.load(open(f'{glove_dir}/27B.50_words.pkl', 'rb'))
        word2idx = pickle.load(open(f'{glove_dir}/27B.50_idx.pkl', 'rb'))

        glove = {w: vectors[word2idx[w]] for w in words}
        
        # Embedding matrix : 
        matrix_len = self.dataset.voc_len
        weights_matrix = np.zeros((matrix_len, self.embedding_dim))
        words_found = 0

        for i, word in enumerate(self.dataset.vocab):
            try: 
                weights_matrix[i] = glove[word]
                words_found += 1
            except KeyError:
                weights_matrix[i] = np.random.normal(scale=0.6, size=(self.embedding_dim, ))
                
        return torch.tensor(weights_matrix)
        
    
if(__name__=='__main__'):
    loaders = Loader()
    train, _, test = loaders.get_data()