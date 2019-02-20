import tweepy
import pandas as pd
from nltk.sentiment.vader import SentimentIntensityAnalyzer


#authentication variables from twitter
consumer_key = ''
consumer_secret = ''
access_token = ''
access_token_secret = ''

# authentication of the twitter variable
auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)

api = tweepy.API(auth)

tweets=api.search("Artificial Intelligence",count=200)

data = pd.DataFrame(data=[tweet.text for tweet in tweets], columns=['Tweets'])

display(data.head(10))

print(tweets[0].created_at)

import nltk
nltk.download('vader_lexicon')
sid = SentimentIntensityAnalyzer()


result= []

for index, row in data.iterrows():
  ss = sid.polarity_scores(row["Tweets"])
  result.append(ss)
  
se = pd.Series(result)
data['polarity'] = se.values

display(data.head(100))
