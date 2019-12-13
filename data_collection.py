from twython import Twython 
import pandas as pd
import pickle
from datetime import date 

FIRST = True
SECONDRUN = not FIRST

FIRSTGO = True
# Dict assigning an integer label to each of the topics we have 
#label_dict={topic:i for topic,i in enumerate(topics)}
def saveID(curID): 
    with open('id.txt','w') as f: 
        f.write(str(curID))
def saveData(data): 
    global FIRST 
    df = pd.DataFrame(data, columns = ['Tweet','Label'])
    if FIRST: 
        df.to_csv('dataframe.csv')
        FIRST = False
    else:
        with open('dataframe.csv','a',encoding="utf-8") as f:
            df.to_csv(f,header=False)
def loadID (): 
    with open('id.txt','r') as f: 
        curID = f.read()
        curID = int(curID)
    return curID
def saveState(topic,keyword,curNum): 
    with open('curState.txt','w') as f: 
        f.write(str(topic))
        f.write(str(keyword))
        f.write(str(curNum))
def loadState():
    with open('curState.txt','r') as f: 
        curState = f.read() 
    topic = curState[0]
    keyword = curState[1]
    curNum = curState[2:]
    return int(topic), int(keyword), int(curNum)
def loadIterations(): 
    with open('iterations.txt', 'r') as f: 
        iterations = f.read()
    return int(iterations)
def saveIterations(iterations): 
    with open('iterations.txt', 'w') as f: 
        f.write(str(iterations))
    
    
def getSection(numKeyword): 
    dates = ['2019-11-20', '2019-11-19']
    curID = []
    for date in dates: 
        query = {'q': 'e',
            'result_type': 'mixed',
            'count': 1,
            'until': date,
            'lang': 'en'} 
        for status in python_tweets.search(**query)['statuses']:
            curID.append(status['id']) 
    difference = curID[0] - curID[1]
    section = (difference*5)//(5*(numKeyword//100))
    return section 
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
        newDay = str(day - 5)
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

def getQuery(keyword,count,curID,topic,data,pastID): 
    query = {'q': keyword,
    'result_type': 'mixed',
    'count': count,
    'lang': 'en',
    'since_id': pastID,
    'max_id': curID} 
    for status in python_tweets.search(**query)['statuses']:
        data.append([status['text'],int(topic)])
    return data


def getData(queryList,topic,data,requiredNum,curNum,divisor,curID,intKeyword,pastID):
    global SECONDRUN
    global FIRSTGO 
    # set up how much data to retrieve !make sure that all lengths add up for dataframe
    numKeyword = requiredNum//divisor
    if topic == '4':
        numKeyword = requiredNum - curNum
    # access api (note I had to used mixed for result type instead of popular to get the desired amount of tweets)
    if not SECONDRUN: 
        iterations = numKeyword//100
    if SECONDRUN and FIRSTGO: 
        iterations = loadIterations()
        FIRSTGO = False
    else: 
        iterations = numKeyword//100

    overflow = numKeyword%100
    count = 100
    section = getSection(numKeyword)
    while intKeyword < len(queryList):
        while iterations>0: 
            data = getQuery(queryList[intKeyword],count,curID,topic,data,pastID)
            curNum = curNum + count
            pastID = curID
            curID = curID + section
            iterations = iterations-1
            saveData(data)
            saveID(curID)
            saveIterations(iterations)
            saveState(topic,intKeyword,curNum)
            data = []
        
        if overflow is not 0: 
            data = getQuery(queryList[0],overflow,curID,topic,data,pastID)
            curNum = curNum + overflow
            saveData(data)
            saveID(curID)
            saveIterations(iterations)
            saveState(topic,intKeyword,curNum)
            data = []
        iterations = numKeyword//100
        intKeyword = intKeyword + 1
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
queryDict = [queryPolitics, querySports,  queryMovies, queryCompanies,  queryGeneral] 
# make dict that contains each topic
data = []
requiredNum = 200000
divisor = 4

for item in queryDict:
    divisor = divisor + len(item)
lastWeek = getDate()
if not SECONDRUN:
    curID = getID(lastWeek)
    topic = 0
    intKeyword = 0
    curNum = 0
if SECONDRUN: 
    curID = loadID()
    topic,intKeyword,curNum = loadState()
# iterate through queries and retrieve data from twitter
pastID = curID - 2000000
while topic < len(queryDict): 
    data,curNum= getData(queryDict[topic],str(topic),data,requiredNum,curNum,divisor,curID,intKeyword,pastID)
    intKeyword = 0
    topic = topic + 1
    
# for k,v in data.items():
#     print(len(v))

# make data pandas dataframe to facilitate manipulation
# df = pd.DataFrame(data, columns = ['Tweet','Label'])

# save the data for further processing 
# Save in 'data' directory 

# toydata={'tweet':df['4'],'label':np.ones(100)}

# df = pd.DataFrame.from_dict(toydata)

# df.to_csv('dataframe.csv')

"""
pickle.dump(df, open('data/rawTwitterData.pickle', 'wb'))
print(df['0'][6])
"""
