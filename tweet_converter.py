import pickle
import pandas as pd
import numpy as np


""" Removes any words that are @ references to other twitter users, removes 'RT' used for retweets, and removes urls """
def cleanTweets(rawTwitterData):
	cleanedTweets = {}
	for key in rawTwitterData.keys():
		cleanedTweets[key] = []
		for tweet in rawTwitterData[key]:
			cleanedTweet = []
			for word in tweet.split():
				if (word != 'RT' and '@' not in word and "http" not in word):
					cleanedTweet.append(word)

			cleanedTweets[key].append(' '.join(cleanedTweet))

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
		wordVec[vocab.index(word)] = 1.0
		vec.append(wordVec)

	return vec

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
	print("test")