from twython import Twython 
import json 
import pandas as pd
import pickle


# Dict assigning an integer label to each of the topics we have 
#label_dict={topic:i for topic,i in enumerate(topics)}

def getDivider(numTopic, numKeyword): 
    return numTopic*numKeyword
def getQuery(queryList,topic,data,requiredNum,curNum):
    # set up how much data to retrieve !make sure that all lengths add up for dataframe
    divider = getDivider(5,len(queryList))
    count = requiredNum/divider
    if topic is '4':
        count = requiredNum - curNum
    # access api (note I had to used mixed for result type instead of popular to get the desired amount of tweets)
    for keyword in queryList: 
        curNum = curNum + count
        query = {'q': keyword,
         'result_type': 'mixed',
         'count': count,
         'lang': 'en'} 
        for status in python_tweets.search(**query)['statuses']:
            data.append([status['text'],int(topic)])
    return data, curNum


# create object with twitter keys
python_tweets = Twython('W1f186LSrJ28oK0AezosHQncf', 'mr9An5n39GuDZguINAlokrsjtxHKL6mjVHAi00abNANjqelKKi')

# queries list with 5 keywords
queryPolitics =['brexit', 'trump', 'EU', 'trudeau', 'trade war']
querySports = ['habs', 'team', 'basketball', 'win', 'playoff']
queryMovies = ['parasite', 'joker', 'spiderman', 'actor','actress'] 
queryCompanies = ['apple','microsoft','tesla','amazon','google']
# use most common character in english to get general tweets on twitter
queryGeneral =['e']

# make dict of queries [politics = 0, sports = 1, movies = 2, companies = 3, general = 4]
queryDict = {'0': queryPolitics, '1': querySports, '2': queryMovies, '3': queryCompanies, '4': queryGeneral}
# make dict that contains each topic
data = []
requiredNum = 200000
curNum = 0
# iterate through queries and retrieve data from twitter
for key in queryDict: 
    data,curNum= getQuery(queryDict[key],key,data,requiredNum,curNum)
    
# for k,v in data.items():
#     print(len(v))

# make data pandas dataframe to facilitate manipulation
df = pd.DataFrame(data, columns = ['Tweet','Label'])

# save the data for further processing 
# Save in 'data' directory 

# toydata={'tweet':df['4'],'label':np.ones(100)}

# df = pd.DataFrame.from_dict(toydata)

df.to_csv('dataframe.csv')

"""
pickle.dump(df, open('data/rawTwitterData.pickle', 'wb'))
print(df['0'][6])
"""
