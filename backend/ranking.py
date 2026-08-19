"""
==========================================
Instagram Engagement Rate & Ranking
==========================================
"""

import numpy as np


def calculate_engagement_rate(predicted_likes, comments, followers):
    """
    Calculate Engagement Rate
    Formula:
    ((Likes + Comments) / Followers) * 100
    """

    if followers == 0:
        return 0

    engagement_rate = (
        (predicted_likes + comments) / followers
    ) * 100

    return round(engagement_rate, 2)


def calculate_rank(engagement_rate):
    """
    Assign rank based on engagement rate.
    """

    if engagement_rate >= 10:
        return "⭐⭐⭐⭐⭐ Excellent"

    elif engagement_rate >= 7:
        return "⭐⭐⭐⭐ Very Good"

    elif engagement_rate >= 4:
        return "⭐⭐⭐ Good"

    elif engagement_rate >= 2:
        return "⭐⭐ Average"

    else:
        return "⭐ Needs Improvement"


def get_result(predicted_likes, comments, followers):
    """
    Returns likes, engagement rate and rank.
    """

    engagement_rate = calculate_engagement_rate(
        predicted_likes,
        comments,
        followers
    )

    rank = calculate_rank(engagement_rate)

    return {
        "Predicted Likes": round(predicted_likes),
        "Comments": comments,
        "Followers": followers,
        "Engagement Rate": engagement_rate,
        "Rank": rank
    }


# ==========================================
# Testing
# ==========================================

if __name__ == "__main__":

    predicted_likes = 2850

    comments = 120

    followers = 15000

    result = get_result(
        predicted_likes,
        comments,
        followers
    )

    print("\n========== RESULT ==========\n")

    for key, value in result.items():
        print(f"{key}: {value}")