"""
==========================================
Instagram Engagement Predictor
Exploratory Data Analysis
==========================================
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# -----------------------------
# Paths
# -----------------------------

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_PATH = os.path.join(
    BASE_DIR,
    "dataset",
    "processed_instagram.csv"
)

REPORT_PATH = os.path.join(
    BASE_DIR,
    "reports"
)

os.makedirs(REPORT_PATH, exist_ok=True)

# -----------------------------
# Load Data
# -----------------------------

df = pd.read_csv(DATA_PATH)

print("="*50)
print("EDA STARTED")
print("="*50)

print(df.shape)
print(df.head())

# -----------------------------
# Correlation
# -----------------------------

numeric = df.select_dtypes(include=["int64","float64"])

corr = numeric.corr()

plt.figure(figsize=(12,8))

sns.heatmap(
    corr,
    annot=True,
    cmap="coolwarm"
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        REPORT_PATH,
        "correlation_heatmap.png"
    )
)

plt.close()

print("Correlation Heatmap Saved")

# -----------------------------
# Likes Distribution
# -----------------------------

plt.figure(figsize=(8,6))

sns.histplot(
    df["likes"],
    bins=50
)

plt.title("Likes Distribution")

plt.savefig(
    os.path.join(
        REPORT_PATH,
        "likes_distribution.png"
    )
)

plt.close()

print("Likes Distribution Saved")

# -----------------------------
# Followers vs Likes
# -----------------------------

plt.figure(figsize=(8,6))

sns.scatterplot(
    x=df["followers"],
    y=df["likes"]
)

plt.savefig(
    os.path.join(
        REPORT_PATH,
        "followers_vs_likes.png"
    )
)

plt.close()

print("Scatter Plot Saved")

print("="*50)
print("EDA COMPLETED")
print("="*50)