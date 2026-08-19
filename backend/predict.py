import os
import joblib
import numpy as np
import pandas as pd

# ===========================
# Load Model
# ===========================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "best_model.pkl"
)

model = joblib.load(MODEL_PATH)

print("Best model loaded successfully!")

# ===========================
# Feature Order
# ===========================

features = [

    "post_type",

    "comments",

    "following",

    "followers",

    "num_posts",

    "is_business_account",

    "lang",

    "description_category",

    "description_grade",

    "image_grade",

    "follower_following_ratio",

    "caption_length",

    "word_count",

    "hashtag_count",

    "mention_count"

]

# ===========================
# Example Input
# ===========================

sample = {

    "post_type": 1,

    "comments": 120,

    "following": 650,

    "followers": 15000,

    "num_posts": 520,

    "is_business_account": 1,

    "lang": 0,

    "description_category": 2,

    "description_grade": 8,

    "image_grade": 9,

    "follower_following_ratio": 23.07,

    "caption_length": 145,

    "word_count": 28,

    "hashtag_count": 7,

    "mention_count": 2

}

# ===========================
# Prediction
# ===========================

input_df = pd.DataFrame([sample])

prediction_log = model.predict(input_df)[0]

predicted_likes = np.expm1(prediction_log)

print("\nPredicted Likes:", round(predicted_likes))