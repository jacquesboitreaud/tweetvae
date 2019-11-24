# -*- coding: utf-8 -*-
"""
Created on Sun Nov 24 14:58:32 2019

@author: jacqu

Loads GloVe word embeddings 
"""

import bcolz
import numpy as np
import pickle

glove_path = 'C:/Users/jacqu/Documents/GitHub/tweetvae/data'

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

"""
# Load vectors 
vectors = bcolz.open(f'{glove_path}/6B.50.dat')[:]
words = pickle.load(open(f'{glove_path}/6B.50_words.pkl', 'rb'))
word2idx = pickle.load(open(f'{glove_path}/6B.50_idx.pkl', 'rb'))

glove = {w: vectors[word2idx[w]] for w in words}

# glove['the'] => vector for 'the'

#################################################################
# Embed dataset vocabulary #####################################
#################################################################

# FOr each word in dataset vocabulary: 

matrix_len = len(target_vocab)
weights_matrix = np.zeros((matrix_len, 50))
words_found = 0

for i, word in enumerate(target_vocab):
    try: 
        weights_matrix[i] = glove[word]
        words_found += 1
    except KeyError:
        weights_matrix[i] = np.random.normal(scale=0.6, size=(emb_dim, ))
"""