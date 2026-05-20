# analysis.py

import matplotlib.pyplot as plt
import seaborn as sns


def univariate_analysis(df):

    # Age Distribution
    plt.figure(figsize=(6,4))

    df['Age'].hist(bins=20)

    plt.title("Age Distribution")
    plt.xlabel("Age")
    plt.ylabel("Count")

    plt.show()

    # Income Boxplot
    plt.figure(figsize=(6,4))

    sns.boxplot(y=df['Annual Income (k$)'])

    plt.title("Income Boxplot")

    plt.show()


def bivariate_analysis(df, features):

    # Scatter Plot
    plt.figure(figsize=(6,4))

    sns.scatterplot(
        x='Annual Income (k$)',
        y='Spending Score (1-100)',
        data=df
    )

    plt.title("Income vs Spending")

    plt.show()

    # Pair Plot
    sns.pairplot(df[features])

    plt.show()

    # Correlation Heatmap
    corr = df[features].corr()

    plt.figure(figsize=(6,4))

    sns.heatmap(
        corr,
        annot=True,
        cmap='coolwarm'
    )

    plt.title("Correlation Heatmap")

    plt.show()