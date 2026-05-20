# app.py

from data_collection import load_data

from preprocessing import preprocess_data

from analysis import (
    univariate_analysis,
    bivariate_analysis
)

from clustering import (
    elbow_method,
    silhouette_scores,
    train_model
)

from visualization import visualize_clusters



# Load Dataset


file_path = "Mall_Customers.csv"

df = load_data(file_path)



# Preprocessing


df, X_scaled, features = preprocess_data(df)



# Exploratory Analysis

univariate_analysis(df)

bivariate_analysis(df, features)



# Model Selection


elbow_method(X_scaled)

silhouette_scores(X_scaled)



# Final Model


model, labels = train_model(
    X_scaled,
    n_clusters=5
)

df['Cluster'] = labels



# Business Segment Names

segment_names = {
    0: "High-Value Customers",
    1: "Budget Customers",
    2: "Rare Buyers",
    3: "Premium Customers",
    4: "Young Spenders"
}

df['Segment_Name'] = (
    df['Cluster']
    .map(segment_names)
)



# Cluster Summary


summary = df.groupby('Segment_Name')[
    features
].mean()

print("\nCluster Summary:\n")

print(summary)


# Visualization


visualize_clusters(df, X_scaled)