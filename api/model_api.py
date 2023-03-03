# 1. Library imports
import pandas as pd 
from sklearn.ensemble import RandomForestClassifier
from pydantic import BaseModel
import joblib
import pickle
import tensorflow as tf
from keras.preprocessing.text import Tokenizer
from keras.models import load_model


# 2. Class which describes a single flower measurements
class Tweet(BaseModel):
    tweet_path: str


# 3. Class for training the model and making predictions
class SenimentModel:
    # 6. Class constructor, loads the dataset and loads the model
    #    if exists. If not, calls the _train_model method and 
    #    saves the model
    def __init__(self, tweet_path: Tweet):
        self._tweet_path = tweet_path
        self._model_path = '../model/saved_model/blstm_model'
        self._tokenizer_path = '../model/saved_tokenizer/tokenizer.pickle'
        try:
            # Load the saved tokenizer
            self.tokenizer = Tokenizer()
            with open(self._tokenizer_path, 'rb') as handle:
                self.tokenizer = pickle.load(handle)
            # Load the saved model
            self.model = load_model(self._model_path)

        except Exception as e: print(e)

    def _scrapp_tweet(self):
        self.tweet = tweet

    def _preprocess_tweet(self):
        tweet = _scrapp_tweet()

    def predict(self):
        
        prediction = self.model.predict(tweet)
        probability = self.model.predict_proba(data_in).max()
        return prediction[0], probability