from twython import Twython 
import json 
import pandas as pd
import pickle

def getQuery(queryList,topic,data):
    # set up how much data to retrieve !make sure that all lengths add up for dataframe
    count = 20
    if topic is 'general':
        count = 100
    # access api (note I had to used mixed for result type instead of popular to get the desired amount of tweets)
    for keyword in queryList: 
        query = {'q': keyword,
         'result_type': 'mixed',
         'count': count,
         'lang': 'en'}
        for status in python_tweets.search(**query)['statuses']:
            data[topic].append(status['text'])
    return data 


# create object with twitter keys
python_tweets = Twython('W1f186LSrJ28oK0AezosHQncf', 'mr9An5n39GuDZguINAlokrsjtxHKL6mjVHAi00abNANjqelKKi')

# queries list with 5 keywords
queryPolitics =['brexit', 'trump', 'EU', 'trudeau', 'trade war']
querySports = ['habs', 'team', 'basketball', 'win', 'playoff']
queryMovies = ['parasite', 'joker', 'spiderman', 'actor','actress'] 
queryCompanies = ['apple','microsoft','tesla','amazon','google']
# use most common character in english to get general tweets on twitter
queryGeneral =['e']

# make dict of queries
queryDict = {'politics': queryPolitics, 'sports': querySports, 'movies': queryMovies, 'companies': queryCompanies, 'general': queryGeneral}
# make dict that contains each topic
data = {'politics': [], 'sports': [], 'movies': [], 'companies': [], 'general': []}

# iterate through queries and retrieve data from twitter
for key in queryDict: 
    data = getQuery(queryDict[key],key,data)

# make data pandas dataframe to facilitate manipulation
df = pd.DataFrame(data)

# save the data for further processing
pickle.dump(df, open('rawTwitterData.pickle', 'wb'))
print(df['politics'][6])
