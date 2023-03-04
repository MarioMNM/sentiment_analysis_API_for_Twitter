# 1. Library imports
import uvicorn
from fastapi import FastAPI, HTTPException
from api.model_api import SearchedTweets, SenimentModel



# 2. Create app and model objects
app = FastAPI()
model = SenimentModel()



class NoMatchException(HTTPException):
    def __init__(self, tweets: SearchedTweets):
        super().__init__(status_code=404, detail=f"No matches found for tweets with attributes: '{tweets}'")


class PredictionErrorException(HTTPException):
    def __init__(self, error_message: str):
        detail = f"Error occurred during prediction using BLSTM model : {error_message}"
        super().__init__(status_code=500, detail=detail)



# 3. Expose the prediction functionality, make a prediction from the passed
#    JSON data and return the predicted flower species with the confidence
@app.post('/predict')
async def predict_sentiment(tweets: SearchedTweets):
    try:
        tweets_df = model.predict(
            tweets.topic_name, tweets.username, tweets.date_init, tweets.date_end, tweets.limit_number_search
        )
    
    except Exception as e:
        raise PredictionErrorException(error_message=str(e))

    if len(tweets_df) == 0:
        raise NoMatchException(tweets)
    
    tweets_json = tweets_df.to_json()
    return tweets_json



# 4. Run the API with uvicorn
#    Will run on http://127.0.0.1:8000
if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)
