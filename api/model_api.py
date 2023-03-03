# 1. Library imports
import pandas as pd
import numpy as np
from pydantic import BaseModel
import joblib
import pickle
import re
import string
import snscrape.modules.twitter as sntwitter
from datetime import datetime, timedelta
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model


# 2. Class which describes a single flower measurements
class SearchedTweets(BaseModel):
    topic_name: str
    username: str
    date_init: str
    date_end: str
    limit_number_search: int
    


# 3. Class for training the model and making predictions
class SenimentModel:
    # 6. Class constructor, loads the dataset and loads the model
    #    if exists. If not, calls the _train_model method and 
    #    saves the model
    def __init__(self):
        self._model_path = './model/saved_model/blstm_model'
        self._tokenizer_path = './model/saved_tokenizer/tokenizer.pickle'

        try:
            # Load the saved tokenizer
            self.tokenizer = Tokenizer()
            with open(self._tokenizer_path, 'rb') as handle:
                self.tokenizer = pickle.load(handle)
            # Load the saved model
            self.model = load_model(self._model_path)

        except Exception as e: print(e)

    def _scrapp_tweet(self, topic_name=None, username=None, date_init=None, date_end=None, limit_number_search=None):
        # Creating list to append tweet data to
        attributes_container = []

        if not limit_number_search: limit_number_search = 100
        if username: username = 'from:' + username
        if date_init: date_init = 'since:' + date_init
        if date_end: 
            # convert the string to a datetime object
            date = datetime.strptime(date_end, '%Y-%m-%d')

            # add one day
            new_date = date + timedelta(days=1)

            # convert the new date back to a string
            new_date_str = datetime.strftime(new_date, '%Y-%m-%d')
            
            date_end = 'until:' + new_date_str

        list_kwords = [username, topic_name, date_init, date_end]

        search_sentence = " ".join([s for s in list_kwords if s])
        # Using TwitterSearchScraper to scrape data and append tweets to list
        for i,tweet in enumerate(sntwitter.TwitterSearchScraper(search_sentence).get_items()):
            if i > (limit_number_search - 1):
                break
            attributes_container.append([tweet.user.username, tweet.date, tweet.likeCount, tweet.sourceLabel, tweet.content])
    
        # Creating a dataframe to load the list
        tweets_df = pd.DataFrame(attributes_container, columns=["User", "Date Created", "Number of Likes", "Source of Tweet", "Tweet"])
        # Applying the cleaning function to both test and training datasets
        tweets_df["Tweet"] = tweets_df["Tweet"].apply(lambda x: clean_text(x))
        # Applying the function to both test and training datasets
        tweets_df["Tweet"] = tweets_df["Tweet"].apply(lambda x: remove_emoji(x))
        return tweets_df[["Date Created", "Number of Likes", "Tweet"]]


    def _preprocess_tweet(self, tweets_df: pd.DataFrame):
        tweets = tweets_df['Tweet'].values

        input_sequence = self.tokenizer.texts_to_sequences(tweets)
        padded_sequence = pad_sequences(input_sequence, maxlen=280, truncating='post')
        return padded_sequence


    def predict(self, topic_name=None, username=None, date_init=None, date_end=None, limit_number_search=None):
        tweets_df = self._scrapp_tweet(topic_name, username, date_init, date_end, limit_number_search)
        processed_tweet = self._preprocess_tweet(tweets_df)

        # predict the sentiment probabilities
        sentiment_probs = self.model.predict(processed_tweet)

        # create a new column with rounded values
        tweets_df['Probability'] = sentiment_probs
        # create a new column with 'Positive' or 'Negative' values
        tweets_df['Sentiment'] = tweets_df['Probability'].apply(lambda x: 'Positive' if round(x) == 1 else 'Negative')

        return tweets_df[["Date Created", "Number of Likes", "Tweet", "Sentiment", "Probability"]]
    
def clean_text(text):
    '''Make text lowercase, remove text in square brackets,remove links,remove punctuation
    and remove words containing numbers.'''
    text = text.lower()
    text = re.sub("@[A-Za-z0-9_]+","", text)
    text = re.sub("#[A-Za-z0-9_]+","", text)
    text = re.sub('\[.*?\]', '', text)
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)
    return text

def remove_emoji(text):
    emoji_pattern = re.compile("["
                           u"\U0001F600-\U0001F64F"  # emoticons
                           u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                           u"\U0001F680-\U0001F6FF"  # transport & map symbols
                           u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           u"\U00002702-\U000027B0"
                           u"\U000024C2-\U0001F251"
                           "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', text)