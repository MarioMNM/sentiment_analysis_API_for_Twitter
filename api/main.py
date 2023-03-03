# 1. Library imports
import uvicorn
from fastapi import FastAPI
from api.model_api import SearchedTweets, SenimentModel

# 2. Create app and model objects
app = FastAPI()
model = SenimentModel()

# 3. Expose the prediction functionality, make a prediction from the passed
#    JSON data and return the predicted flower species with the confidence
@app.post('/predict')
def predict_sentiment(tweets: SearchedTweets):
    tweets_df = model.predict(
        tweets.topic_name, tweets.username, tweets.date_init, tweets.date_end, tweets.limit_number_search
    )
    
    tweets_json = tweets_df.to_json()
    return tweets_json


# 4. Run the API with uvicorn
#    Will run on http://127.0.0.1:8000
if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)