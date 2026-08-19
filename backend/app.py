from flask import Flask, render_template, request
import os
import joblib
import numpy as np
import pandas as pd
import plotly.express as px

from ranking import calculate_engagement_rate, calculate_rank

# ==========================================================
# Flask App
# ==========================================================

app = Flask(
    __name__,
    template_folder="../frontend/templates",
    static_folder="../frontend/static"
)

# ==========================================================
# Project Paths
# ==========================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Load trained model
MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "best_model.pkl"
)

model = joblib.load(MODEL_PATH)

# Load feature importance
FEATURE_PATH = os.path.join(
    BASE_DIR,
    "reports",
    "feature_importance.csv"
)

feature_df = pd.read_csv(FEATURE_PATH)

# ==========================================================
# Home Page
# ==========================================================

@app.route("/")
def home():
    return render_template("index.html")


# ==========================================================
# Prediction
# ==========================================================

@app.route("/predict", methods=["POST"])
def predict():

    try:

        # -------------------------
        # Read Form Data
        # -------------------------

        post_type = int(request.form["post_type"])
        comments = int(request.form["comments"])
        following = int(request.form["following"])
        followers = int(request.form["followers"])
        num_posts = int(request.form["num_posts"])
        is_business_account = int(request.form["is_business_account"])
        lang = int(request.form["lang"])
        description_category = int(request.form["description_category"])
        description_grade = float(request.form["description_grade"])
        image_grade = float(request.form["image_grade"])
        caption_length = int(request.form["caption_length"])
        word_count = int(request.form["word_count"])
        hashtag_count = int(request.form["hashtag_count"])
        mention_count = int(request.form["mention_count"])

        # -------------------------
        # Feature Engineering
        # -------------------------

        if following == 0:
            ratio = followers
        else:
            ratio = followers / following

        # -------------------------
        # Create Input DataFrame
        # -------------------------

        input_df = pd.DataFrame([{

            "post_type": post_type,
            "comments": comments,
            "following": following,
            "followers": followers,
            "num_posts": num_posts,
            "is_business_account": is_business_account,
            "lang": lang,
            "description_category": description_category,
            "description_grade": description_grade,
            "image_grade": image_grade,
            "follower_following_ratio": ratio,
            "caption_length": caption_length,
            "word_count": word_count,
            "hashtag_count": hashtag_count,
            "mention_count": mention_count

        }])

        # -------------------------
        # Predict Likes
        # -------------------------

        prediction_log = model.predict(input_df)[0]

        predicted_likes = int(np.expm1(prediction_log))

        # -------------------------
        # Calculate Engagement
        # -------------------------

        engagement_rate = calculate_engagement_rate(
            predicted_likes,
            comments,
            followers
        )

        # -------------------------
        # Calculate Rank
        # -------------------------

        rank = calculate_rank(
            engagement_rate
        )

        # -------------------------
        # Feature Importance Chart
        # -------------------------

        fig = px.bar(

            feature_df,

            x="Importance",

            y="Feature",

            orientation="h",

            color="Importance",

            title="Feature Importance"

        )

        fig.update_layout(

            template="plotly_white",

            height=500

        )

        feature_chart = fig.to_html(full_html=False)

        # -------------------------
        # Return Results
        # -------------------------

        return render_template(

            "index.html",

            predicted_likes=predicted_likes,

            engagement_rate=round(engagement_rate, 2),

            rank=rank,

            feature_chart=feature_chart

        )

    except Exception as e:

        return render_template(

            "index.html",

            error=str(e)

        )


# ==========================================================
# Run Flask App
# ==========================================================

if __name__ == "__main__":

    app.run(
        debug=True,
        port=5001
    )