# 🌸 Iris Flower Prediction System

A Machine Learning mini-project that predicts the species of an Iris flower based on its sepal and petal measurements.

The project uses a Random Forest classification model and exposes the prediction through a FastAPI REST API. A simple HTML, CSS and JavaScript frontend allows users to enter flower measurements and receive a prediction with confidence.

---

## 📌 Project Overview

The Iris Flower Prediction System takes four measurements from the user:

- Sepal Length
- Sepal Width
- Petal Length
- Petal Width

The trained Machine Learning model predicts one of the following Iris species:

- Setosa
- Versicolor
- Virginica

The prediction is displayed on the frontend along with the model's confidence.

---

## 🎯 Objective

The main objective of this project is to demonstrate an end-to-end Machine Learning application.

The project covers:

1. Loading a Machine Learning dataset
2. Training a classification model
3. Saving the trained model
4. Loading the saved model
5. Creating a REST API
6. Connecting a frontend to the API
7. Displaying the prediction to the user

---

## 🧠 Machine Learning Model

### Dataset

The project uses the Iris dataset provided by Scikit-learn.

The dataset contains measurements of Iris flowers belonging to three different species.

### Input Features

The model uses four features:

| Feature | Description |
|---|---|
| Sepal Length | Length of the sepal in centimeters |
| Sepal Width | Width of the sepal in centimeters |
| Petal Length | Length of the petal in centimeters |
| Petal Width | Width of the petal in centimeters |

### Output

The model predicts:

```text
Setosa
Versicolor
Virginica