import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from preprocess import load_data, preprocess_data

def train_model(model_name, data_path="../data/heart_dataset.csv"):
    # Load and preprocess data
    df = load_data(data_path)
    df = preprocess_data(df)

    # Split into features (X) and target (y)
    X = df.drop("target", axis=1)  # assumes "target" column is the label
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Select model
    if model_name == "logistic":
        model = LogisticRegression(max_iter=1000)
    elif model_name == "decision_tree":
        model = DecisionTreeClassifier()
    elif model_name == "random_forest":
        model = RandomForestClassifier()
    else:
        raise ValueError("Model not supported!")

    # Train
    model.fit(X_train, y_train)

    # Predict
    y_pred = model.predict(X_test)

    # Evaluate
    acc = accuracy_score(y_test, y_pred)
    print(f"{model_name} Accuracy: {acc:.2f}")

    return model

if __name__ == "__main__":
    # Example: Train logistic regression
    train_model("logistic")
