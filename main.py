from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.openapi.utils import get_openapi
from sentimentAnalysis import getModels

logisticModel, bayesModel = getModels()

class SentimentResponse(BaseModel):
    originalText: str
    predictedSentiment: str

app = FastAPI() # uvicorn main:app

@app.get("/predictSentiment/", response_model=SentimentResponse)
async def predictSentiment(text: str):
    '''Returns the predicted sentiment ("positive" or "negative") of a given string (text)'''
    return {"originalText": text, "predictedSentiment": logisticModel.predict(text)}

def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema
    openapi_schema = get_openapi(
        title="Sentiment Analysis API",
        version="1.0.0",
        description="Documentation for a simple API that takes english text and provides a predicted sentiment (positive/negative) for it",
        routes=app.routes,
    )
    app.openapi_schema = openapi_schema
    return app.openapi_schema

app.openapi = custom_openapi