# clustering.py

import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


def elbow_method(X_scaled):

    wcss = []

    for k in range(1, 11):

        model = KMeans(
            n_clusters=k,
            random_state=42
        )

        model.fit(X_scaled)

        wcss.append(model.inertia_)

    # Plot Elbow Curve
    plt.figure(figsize=(6,4))

    plt.plot(
        range(1, 11),
        wcss,
        marker='o'
    )

    plt.title("Elbow Method")
    plt.xlabel("Clusters")
    plt.ylabel("WCSS")

    plt.show()


def silhouette_scores(X_scaled):

    print("\nSilhouette Scores:\n")

    for k in range(2, 11):

        model = KMeans(
            n_clusters=k,
            random_state=42
        )

        labels = model.fit_predict(X_scaled)

        score = silhouette_score(
            X_scaled,
            labels
        )

        print(f"K={k} : {score:.3f}")


def train_model(X_scaled, n_clusters=5):

    model = KMeans(
        n_clusters=n_clusters,
        random_state=42
    )

    labels = model.fit_predict(X_scaled)

    return model, labels