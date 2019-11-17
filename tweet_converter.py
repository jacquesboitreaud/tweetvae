import pickle
import pandas as pd
import numpy as np
from tqdm import tqdm
import re

""" 
I modified some functions to handle the tweet dataframe with columns (tweet, label)
instead of having several keys in the dict
"""


""" Removes any words that are @ references to other twitter users, removes 'RT' used for retweets, and removes urls; keeps only words that contain alphanumerics characters and basic punctuation """
def cleanTweets(rawTwitterData):
    """ Takes dataframe (tweet,label). Returns same with clean tweets """
    cleanedTweets = {'tweet':[], 'label':[],'len':[]}
    for i,row in tqdm(rawTwitterData.iterrows()):
        cleanedTweet = []
        tweet,label = row['tweet'], row['label']
        for word in tweet.split():
            if (word != 'RT' and "http" not in word and bool(re.match(r'[\w.,?!\%/ \-()]+$', word))):
                cleanedTweet.append(word)
                
        length=len(cleanedTweet)   
        cleanedTweets['tweet'].append(' '.join(cleanedTweet))
        cleanedTweets['label'].append(label)
        cleanedTweets['len'].append(length)

    return pd.DataFrame(cleanedTweets)

""" Generates the vector of possible vocabulary words based on the sampled tweets, returned as a list of words """
def generateDictionaryList(cleanedTwitterData):
	sample_vocab = []

	for key in cleanedTwitterData.keys():
		for tweet in cleanedTwitterData[key]:
			for word in tweet.split():
				if (word not in sample_vocab):
					sample_vocab.append(word)

	return sample_vocab

""" Converts a tweet, passed as a string, into a 2D matrix with each row being a one-hot representation of each of the words """
def tweetToVec(tweet, vocab):
	vec = []
	vocabLength = len(vocab)

	for word in tweet.split():
		wordVec = np.zeros(vocabLength).tolist()
		wordVec[vocab.index(word)] = 1
		vec.append(wordVec)

	return vec

def padded_tweetToVec(tweet,vocab, pad_size):
    # Returns tweet to vec vector padded with zeros at the end it has fixed size (pad_size)
    vocabLength = len(vocab)
    tvec = np.zeros((pad_size,vocabLength))
    
    for i,word in enumerate(tweet.split()):
        tvec[i,vocab.index(word)] =1
        
    return tvec

""" Converts a vector of one-hot encoded word vectors back into a string representing a tweet """
def vecToTweet(tweetVec, vocab):
	constructedTweet = []
	for wordVec in tweetVec:
		constructedTweet.append(vocab[wordVec.index(1.0)])

	return ' '.join(constructedTweet)


""" Loads the collected tweets, encodes them with one-hot encoding, and returns the results """
def getVectorizedTweets():
	rawData = pickle.load(open('rawTwitterData.pickle', 'rb'))
	cleanedData = cleanTweets(rawData)
	sampleVocab = generateDictionaryList(cleanedData)

	vectorizedTweets = {}

	for key in cleanedData.keys():
		vectorizedTweets[key] = []
		for atweet in cleanedData[key]:
			vectorizedTweets[key].append(tweetToVec(atweet, sampleVocab))

	return vectorizedTweets


if __name__ == '__main__':
	