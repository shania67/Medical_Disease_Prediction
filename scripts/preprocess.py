import pandas as pd
from sklearn.preprocessing import StandardScaler

def load_data(file_path):
    """Load dataset from CSV file"""
    return pd.read_csv(file_path)

def preprocess_data(df):
    """Handle missing values and scale numeric columns"""
    # Fill missing values with column mean
    df = df.fillna(df.mean(numeric_only=True))
    
    # Scale numeric features
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
    scaler = StandardScaler()
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
    
    return df

if __name__ == "__main__":
    # Example usage (you can change filename later)
    data = load_data("../data/heart_dataset.csv")  # your dataset
    processed = preprocess_data(data)
    print(processed.head())

