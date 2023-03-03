# Sentiment Analysis API for Twitter with a Deep Learnig model and FastAPI

This project implements an API with a deep learning model (Bidirectional LSTM) for sentiment analysis of tweets by topic, username, and/or date. The model was trained on a large dataset of tweets (around 1.6 million), and it can classify each tweet as positive or negative.

## Installation

To use this API, you need to have Python 3 installed on your system. You can then install the required packages by running:

```
pip install -r requirements.txt
```

This will install all the necessary packages, including TensorFlow, FastAPI and Uvicorn.

## Usage

To start the API server, run:

```
uvicorn api.main:app --reload
```


This will start the FastAPI server, which listens on `http://127.0.0.1:8000/`. I strongly recommend to visit the following page wich offers an interactive UI of the API: `http://127.0.0.1:8000/docs`

You can then send requests to the API using the following attributes:

- `topic_name`: Returns the sentiment analysis of the latest tweets about a given topic.
- `username`: Returns the sentiment analysis of the latest tweets by a given user.
- `date_init`: Returns the sentiment analysis of the tweets posted since a given date.
- `date_end`: Returns the sentiment analysis of the tweets posted until a given date.
- `limit_number_search`: Limits the number of tweets scraped for the sentiment analysis.

The API will return a JSON response with the sentiment analysis results.

## License

This project is licensed under the MIT License. You can use it for any purpose, including commercial projects. However, please credit the original author and provide a link to the original repository.
