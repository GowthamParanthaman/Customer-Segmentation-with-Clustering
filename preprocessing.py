# preprocessing.py

import pandas as pd

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler


def preprocess_data(df):

    
    # Handle Missing Values

    df.fillna(df.mean(numeric_only=True), inplace=True)

    
    # Remove Duplicates

    df.drop_duplicates(inplace=True)

    
    # Encode Gender Column
    
    encoder = LabelEncoder()

    df['Gender'] = encoder.fit_transform(df['Gender'])

    
    # Feature Selection
    
    features = [
        'Age',
        'Annual Income (k$)',
        'Spending Score (1-100)'
    ]

    X = df[features]

    
    # Feature Scaling
    
    scaler = StandardScaler()

    X_scaled = scaler.fit_transform(X)

    return df, X_scaled, features