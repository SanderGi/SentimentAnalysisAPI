# SentimentAnalysisAPI
A simple API that takes English text and provides a predicted sentiment (positive/negative) for it. It is a python server using Fast API that predicts sentiments using logistic regression. The documentation uses Swagger UI (mostly the default Fast API stuff, but also some custom descriptions etc.).

## How to Use
* First download the files and uvicorn (to host the Fast API server locally)
* Install dependencies (can be found in requirements.txt and installed using pip)
* Then run "uvicorn main:app" from the terminal to start the local server (from the directory of the files in this github)
* Documentation can be found here when hosted locally: http://127.0.0.1:8000/docs#/
* GET requests can then be made to "http://127.0.0.1:8000/predictSentiment/" where the "text" query parameter is the url encoded english text to get a predicted sentiment of (response is json): <br>
![image](https://user-images.githubusercontent.com/97496861/179452562-7c5327f3-07a2-479c-b01a-d05652a7556d.png)
* The "predictedSentiment" can either be "positive" or "negative"

## Files
* ExploreData.ipynb -- Notebook used to explore data and NLP models
* sentimentAnalysis.py -- Python file that contains all the classes for handling the data, models, inference, etc. using an OOP approach
* airline_sentiment_analysis.csv -- Training data from TrueFoundry ML Internship Project
* *.pickle -- Pretrained models (created by sentimentAnalysis.py if they are not provided)
* .deta and requirements.txt -- Stuff to host the server using Deta Micros (still in progress)
* main.py -- Where the magic happens and the server is created

## Next steps
* Host using Deta or Heroku -- I am currently in the process of hosting a public endpoint for the API but ran into problems with incompatible dependencies for Deta and haven't had time to fix this yet
* Make it possible to choose different models with a query parameter
* Possibly a "feature importance" feature that highlights what words and phrases in the given english text contributed in what ways to the predicted sentiment (NaiveBayes model is useful for this since it is probabilistic)
