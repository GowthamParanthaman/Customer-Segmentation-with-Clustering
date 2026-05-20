import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA


# ---------------------------------------------------
# PAGE TITLE
# ---------------------------------------------------

st.title("Customer Segmentation App")


# ---------------------------------------------------
# FILE UPLOAD
# ---------------------------------------------------

uploaded_file = st.file_uploader(
    "Upload CSV File",
    type=["csv"]
)


# ---------------------------------------------------
# MAIN PROCESS
# ---------------------------------------------------

if uploaded_file:

    # Load dataset
    df = pd.read_csv(uploaded_file)

    st.subheader("Dataset Preview")

    st.write(df.head())


    # ---------------------------------------------------
    # PREPROCESSING
    # ---------------------------------------------------

    # Encode Gender
    if 'Gender' in df.columns:

        encoder = LabelEncoder()

        df['Gender'] = encoder.fit_transform(df['Gender'])


    # Select Features
    features = [
        'Age',
        'Annual Income (k$)',
        'Spending Score (1-100)'
    ]

    X = df[features]


    # Scaling
    scaler = StandardScaler()

    X_scaled = scaler.fit_transform(X)


    # ---------------------------------------------------
    # TRAIN KMEANS
    # ---------------------------------------------------

    model = KMeans(
        n_clusters=5,
        random_state=42
    )

    df['Cluster'] = model.fit_predict(X_scaled)


    # ---------------------------------------------------
    # SEGMENT NAMES
    # ---------------------------------------------------

    segment_names = {
        0: "High Value",
        1: "Budget Customers",
        2: "Premium Customers",
        3: "Young Spenders",
        4: "Rare Buyers"
    }

    df['Segment_Name'] = df['Cluster'].map(segment_names)


    # ---------------------------------------------------
    # SHOW RESULTS
    # ---------------------------------------------------

    st.subheader("Clustered Customers")

    st.write(
        df[
            [
                'CustomerID',
                'Age',
                'Annual Income (k$)',
                'Spending Score (1-100)',
                'Cluster',
                'Segment_Name'
            ]
        ]
    )


    # ---------------------------------------------------
    # CLUSTER SUMMARY
    # ---------------------------------------------------

    st.subheader("Cluster Summary")

    summary = df.groupby('Segment_Name')[
        features
    ].mean()

    st.write(summary)


    # ---------------------------------------------------
    # PCA VISUALIZATION
    # ---------------------------------------------------

    pca = PCA(n_components=2)

    X_pca = pca.fit_transform(X_scaled)

    pca_df = pd.DataFrame({
        'PCA1': X_pca[:, 0],
        'PCA2': X_pca[:, 1],
        'Cluster': df['Cluster']
    })


    # Plot
    fig, ax = plt.subplots(figsize=(8,6))

    sns.scatterplot(
        data=pca_df,
        x='PCA1',
        y='PCA2',
        hue='Cluster',
        palette='Set2',
        s=100,
        ax=ax
    )

    plt.title("Customer Segments")

    st.pyplot(fig)


    # ---------------------------------------------------
    # CLUSTER COUNT
    # ---------------------------------------------------

    st.subheader("Customers Per Cluster")

    fig2, ax2 = plt.subplots(figsize=(6,4))

    sns.countplot(
        x='Cluster',
        data=df,
        palette='Set2',
        ax=ax2
    )

    st.pyplot(fig2)


    # ---------------------------------------------------
    # AVG SPENDING SCORE
    # ---------------------------------------------------

    st.subheader("Average Spending Score")

    avg_spending = df.groupby('Segment_Name')[
        'Spending Score (1-100)'
    ].mean()

    fig3, ax3 = plt.subplots(figsize=(7,4))

    avg_spending.plot(
        kind='bar',
        ax=ax3
    )

    plt.ylabel("Average Spending Score")

    st.pyplot(fig3)