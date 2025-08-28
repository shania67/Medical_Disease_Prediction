import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

from preprocess import load_data, preprocess_data

def train_model(model_type="logistic"):
    """Train a model and return it"""
    # Load and preprocess data
    df = load_data("../data/heart_dataset.csv")
    df = preprocess_data(df)

    # Assume last column is the target (disease/no-disease)
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    # Split into train/test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Select model
    if model_type == "logistic":
        model = LogisticRegression(max_iter=1000)
    elif model_type == "tree":
        model = DecisionTreeClassifier()
    elif model_type == "forest":
        model = RandomForestClassifier()
    else:
        raise ValueError("Invalid model type")

    # Train
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"{model_type} accuracy: {acc:.2f}")

    # Save model
    joblib.dump(model, f"../results/{model_type}_model.pkl")

    return model

if __name__ == "__main__":
    train_model("logistic")
    train_model("tree")
    train_model("forest")
