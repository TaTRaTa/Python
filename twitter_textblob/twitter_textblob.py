import tweepy
from textblob import TextBlob

consumer_key = ''
consumer_secret = ''
access_token = ''
access_token_secret = ''

OAuto = tweepy.OAuthHandler(consumer_key, consumer_secret)
OAuto.set_access_token(access_token, access_token_secret)

api = tweepy.API(OAuto)
public_tweets = api.search('Elon Musk')

for tw in public_tweets:
    print(tw.source + ' ||| ' + tw.text)
    ana = TextBlob(tw.text)
    print(ana.sentiment)
    print(ana.detect_language())
    print(ana.translate(to='bg'))
    print()