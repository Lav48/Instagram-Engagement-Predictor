import os
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

INPUT_PATH = os.path.join(BASE_DIR, "dataset", "processed_instagram.csv")
OUTPUT_PATH = os.path.join(BASE_DIR, "dataset", "featured_instagram.csv")

df = pd.read_csv(INPUT_PATH)

print("Creating New Features...")

# ---------------------------------
# Followers / Following Ratio
# ---------------------------------

df["follower_following_ratio"] = (
    df["followers"] /
    (df["following"] + 1)
)

# ---------------------------------
# Caption Length
# ---------------------------------

if "description" in df.columns:

    df["caption_length"] = (
        df["description"]
        .fillna("")
        .astype(str)
        .apply(len)
    )

# ---------------------------------
# Number of Words
# ---------------------------------

if "description" in df.columns:

    df["word_count"] = (
        df["description"]
        .fillna("")
        .astype(str)
        .apply(lambda x: len(x.split()))
    )

# ---------------------------------
# Number of Hashtags
# ---------------------------------

if "description" in df.columns:

    df["hashtag_count"] = (
        df["description"]
        .fillna("")
        .astype(str)
        .apply(lambda x: x.count("#"))
    )

# ---------------------------------
# Number of Mentions
# ---------------------------------

if "description" in df.columns:

    df["mention_count"] = (
        df["description"]
        .fillna("")
        .astype(str)
        .apply(lambda x: x.count("@"))
    )

print(df.head())

df.to_csv(OUTPUT_PATH, index=False)

print("Feature Engineering Completed")