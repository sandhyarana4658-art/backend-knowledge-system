import os
import joblib
import numpy as np
from sklearn.datasets import load_iris


# Load Iris dataset information
iris = load_iris()

# Path to saved model
MODEL_PATH = os.path.join(
    os.path.dirname(__file__),
    "model.pkl"
)


# Load trained model
model = joblib.load(MODEL_PATH)


def predict_flower(
    sepal_length,
    sepal_width,
    petal_length,
    petal_width
):

    features = np.array([[
        sepal_length,
        sepal_width,
        petal_length,
        petal_width
    ]])

    # Prediction
    prediction = model.predict(features)[0]

    # Prediction probabilities
    probabilities = model.predict_proba(features)[0]

    # Flower name
    flower_name = iris.target_names[prediction]

    # Confidence
    confidence = probabilities[prediction] * 100

    return {
        "prediction": flower_name,
        "confidence": round(confidence, 2)
    }