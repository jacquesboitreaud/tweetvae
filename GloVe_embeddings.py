# -*- coding: utf-8 -*-
"""
Created on Sun Nov 24 14:58:32 2019

@author: jacqu

Makes GloVe word embeddings dictionaries from the raw data downloaded form glove website.
Raw data must be located in glove_path dir. 

"""

import bcolz
import numpy as np
import pickle

if(__name__=='__main__'):
    glove_path = '../data/glove'
    
    words = []
    idx = 0
    word2idx = {}
    vectors = bcolz.carray(np.zeros(1), rootdir=f'{glove_path}/27B.50.dat', mode='w')
    
    with open(f'{glove_path}/glove.27B.50d.txt', 'rb') as f:
        for l in f:
            line = l.decode().split()
            word = line[0]
            words.append(word)
            word2idx[word] = idx
            idx += 1
            vect = np.array(line[1:]).astype(np.float)
            vectors.append(vect)
        
    vectors = bcolz.carray(vectors[:].reshape((1193514, 50)), rootdir=f'{glove_path}/27B.50.dat', mode='w')
    vectors.flush()
    pickle.dump(words, open(f'{glove_path}/27B.50_words.pkl', 'wb'))
    pickle.dump(word2idx, open(f'{glove_path}/27B.50_idx.pkl', 'wb'))