# visualization.py

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.decomposition import PCA


def visualize_clusters(df, X_scaled):

    # -----------------------------
    # PCA
    # -----------------------------
    pca = PCA(n_components=2)

    X_pca = pca.fit_transform(X_scaled)

    # -----------------------------
    # Scatter Plot
    # -----------------------------
    plt.figure(figsize=(8,6))

    sns.scatterplot(
        x=X_pca[:,0],
        y=X_pca[:,1],
        hue=df['Cluster'],
        palette='Set2',
        s=100
    )

    plt.title("Customer Segments")

    plt.show()

    # -----------------------------
    # Cluster Count
    # -----------------------------
    plt.figure(figsize=(6,4))

    sns.countplot(
        x='Cluster',
        data=df
    )

    plt.title("Customers per Cluster")

    plt.show()

    # -----------------------------
    # Average Spending
    # -----------------------------
    avg_spending = df.groupby('Cluster')[
        'Spending Score (1-100)'
    ].mean()

    avg_spending.plot(kind='bar')

    plt.title("Average Spending per Cluster")

    plt.ylabel("Spending Score")

    plt.show()