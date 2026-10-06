from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
import joblib
import os


def train_model():

    # Load Iris dataset
    iris = load_iris()

    X = iris.data
    y = iris.target

    # Create ML model
    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )

    # Train model
    model.fit(X, y)

    # Save trained model
    model_path = os.path.join(
        os.path.dirname(__file__),
        "model.pkl"
    )

    joblib.dump(model, model_path)

    print("Model trained successfully!")
    print(f"Model saved at: {model_path}")


if __name__ == "__main__":
    train_model()