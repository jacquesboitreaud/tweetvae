from twython import Twython 
import pandas as pd
import pickle
from datetime import date 


# Dict assigning an integer label to each of the topics we have 
#label_dict={topic:i for topic,i in enumerate(topics)}
def getID(lastWeek): 
    query = {'q': 'e',
        'result_type': 'mixed',
        'count': 1,
        'until': lastWeek,
        'lang': 'en'} 
    for status in python_tweets.search(**query)['statuses']:
        curID = status['id']
    return curID
def getDate(): 
    today = date.today()
    day = today.strftime("%d")
    month = today.strftime("%m")
    year = today.strftime("%Y")
    day = int(day) 
    if (day>6): 
        newDay = str(day - 6)
        lastWeek =  year + '-' + month + '-' + newDay
    elif month is not '01': 
        newDay = str(30-(5-day))
        newMonth = str(int(month) - 1)
        lastWeek =  year + '-' + newMonth + '-' + newDay
    else: 
        newDay = str(30-(5-day))
        newMonth = '12'
        newYear = str(int(year)-1)
        lastWeek =  newYear + '-' + newMonth + '-' + newDay
    return lastWeek

def getQuery(keyword,count,curID,topic,data): 
    query = {'q': keyword,
    'result_type': 'mixed',
    'count': count,
    'lang': 'en',
    'max_id': curID} 
    for status in python_tweets.search(**query)['statuses']:
        data.append([status['text'],int(topic)])
    return data


def getData(queryList,topic,data,requiredNum,curNum,divisor,curID):
    # set up how much data to retrieve !make sure that all lengths add up for dataframe
    numTopic = requiredNum//divisor
    if topic is '4':
        numTopic = (requiredNum - curNum) + 1
    # access api (note I had to used mixed for result type instead of popular to get the desired amount of tweets)
    iterations = numTopic//100
    overflow = numTopic%100
    count = 100
    while iterations>0:
        for keyword in queryList: 
            data = getQuery(keyword,count,curID,topic,data)
            curID = curID + count
            curNum = curNum + count
        iterations = iterations-1
    if overflow is not 0: 
        data = getQuery(queryList[0],overflow,curID,topic,data)
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
divisor = 4

for key in queryDict:
    divisor = divisor + len(queryDict[key])
lastWeek = getDate()
curID = getID(lastWeek)
# iterate through queries and retrieve data from twitter
for key in queryDict: 
    data,curNum= getData(queryDict[key],key,data,requiredNum,curNum,divisor,curID)
    
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
