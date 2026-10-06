from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from model import predict_flower


# Create FastAPI application
app = FastAPI(
    title="Iris Flower Prediction API",
    description="Machine Learning API for Iris flower classification",
    version="1.0.0"
)


# Enable frontend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)


# Input data format
class FlowerData(BaseModel):

    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float


# Home route
@app.get("/")
def home():

    return {
        "message": "Iris Flower Prediction API is running!",
        "status": "success"
    }


# Prediction route
@app.post("/predict")
def predict(data: FlowerData):

    result = predict_flower(
        data.sepal_length,
        data.sepal_width,
        data.petal_length,
        data.petal_width
    )

    return result