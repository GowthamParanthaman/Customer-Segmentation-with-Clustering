# data_collection.py

import pandas as pd


def load_data(file_path):

    # Load CSV
    df = pd.read_csv(file_path)

    print("Dataset Loaded Successfully\n")

    # First rows
    print(df.head())

    # Shape
    print("\nShape:", df.shape)

    # Columns
    print("\nColumns:")
    print(df.columns)

    # Data types
    print("\nData Types:")
    print(df.dtypes)

    return df


if __name__ == "__main__":

    file_path = "Mall_Customers.csv"

    df = load_data(file_path)