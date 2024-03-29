import pandas as pd
import numpy as np
from tqdm import tqdm
from nltk.stem import WordNetLemmatizer
import re
from nltk.corpus import words
import nltk

""" 
I modified some functions to handle the tweet dataframe with columns (tweet, label)
instead of having several keys in the dict
"""

def clean_dataframe(df):
    """ Takes dataframe as input, filters out nan and inf values if there are. Returns clean df """
    df=df.replace([np.inf, -np.inf], np.nan)
    print('N items before dropping: ', df.shape[0])
    df=df.dropna()
    print('After dropping: ', df.shape[0])
    df=df.reset_index(drop=True)
    return df

""" Removes any words that are @ references to other twitter users, removes 'RT' used for retweets, and removes urls; keeps only words that contain alphanumerics characters and basic punctuation """
def cleanTweets(rawTwitterData):
    """ Takes dataframe (tweet,label). Returns same with clean tweets """
    #Create the lemmatizer and the regex pattern for keeping only alphanumeric characters and spaces
    lemmatizer = WordNetLemmatizer()
    regexPattern = re.compile('[^0-9a-zA-Z ]')
    cleanedTweets = {'tweet':[], 'label':[],'len':[]}
    wordSet = set(words.words())
    for i,row in tqdm(rawTwitterData.iterrows()):
        cleanedTweet = []
        tweet,label = row['tweet'], row['label']
        tweet = nltk.pos_tag(tweet.split())
        for word in tweet:
        	currWord = word[0]
        	if (currWord != 'RT' and currWord != 'rt' and "http" not in currWord and '@' not in currWord):
        		cleanedWord = lemmatizer.lemmatize(regexPattern.sub('',currWord))
        		if (cleanedWord in wordSet or word[1] == 'NNP'):
        			if (word[1] == 'NNP'):
        				cleanedTweet.append('NNP')
        			else:
        				cleanedTweet.append(currWord)
           	#if (currWord != 'RT' and currWord != 'rt' and "http" not in currWord and '@' not in currWord):
            	#remove any non-alphanumeric characters from word and then lemmatize
                
        length=len(cleanedTweet)
        # Filter tweets with less than 7 words remaining
        if(length>6):
            cleanedTweets['tweet'].append(' '.join(cleanedTweet))
            cleanedTweets['label'].append(label)
            cleanedTweets['len'].append(length)

    return pd.DataFrame(cleanedTweets)

""" Takes the cleaned twitter data and replaces any words that occur less than minCount times with 'UNK' """
def removeLowFreqWords(cleanedTwitterData, minCount=10):
    wordCounts = {}
    for atweet in cleanedTwitterData['tweet']:
        for aword in atweet.split():
            if aword in wordCounts:
                wordCounts[aword] += 1
            else:
                wordCounts[aword] = 1

    wordsToKeep = []
    for aword in wordCounts.keys():
        if wordCounts[aword] >= minCount:
            wordsToKeep.append(aword)

    wordSet = set(wordsToKeep)
    filteredTweets = {'tweet':[], 'label':[],'len':[]}

    for atweet in cleanedTwitterData['tweet']:
        sentence = []
        for aword in atweet.split():
            if aword in wordSet:
                sentence.append(aword)
            else:
                sentence.append('UNK')
        filteredTweets['tweet'].append(' '.join(sentence))

    filteredTweets['label'] = cleanedTwitterData['label']
    filteredTweets['len'] = cleanedTwitterData['len']

    return pd.DataFrame(filteredTweets)


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


if __name__ == '__main__':
    """
    #for testing purposes
    testSet = pd.read_csv("test.csv")
    print(testSet['tweet'][1])
    cleaned = removeLowFreqWords(cleanTweets(testSet), 2)

    print(cleaned['tweet'][1])
    cleanedDF = clean_dataframe(cleaned)

    sample_vocab = []
    for tweet in cleanedDF['tweet']:
        if(type(tweet)!=str):
            print('type error, ', type(tweet), tweet)
        for word in tweet.split():
            if (word not in sample_vocab):
                sample_vocab.append(word)
    #print(sample_vocab)
    print(max(cleanedDF['len']))
    """
    
    # To clean the sentiments data 
    df=pd.read_csv('data/sentiments_train.csv')
    df=cleanTweets(df)
    df.to_csv('data/sentiments_train.csv')