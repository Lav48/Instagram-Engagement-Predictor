import os
import warnings
import joblib
import pandas as pd
import numpy as np

warnings.filterwarnings("ignore")

# Machine Learning

from sklearn.model_selection import (
    train_test_split,
    cross_val_score
)

from sklearn.linear_model import (
    LinearRegression,
    Ridge,
    Lasso,
    ElasticNet
)

from sklearn.ensemble import RandomForestRegressor


# Metrics

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# =========================================================
# Project Paths
# =========================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_PATH = os.path.join(
    BASE_DIR,
    "dataset",
    "featured_instagram.csv"
)

MODEL_DIR = os.path.join(
    BASE_DIR,
    "models"
)

REPORT_DIR = os.path.join(
    BASE_DIR,
    "reports"
)

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)

# =========================================================
# Load Dataset
# =========================================================

print("="*60)
print("Loading Dataset...")
print("="*60)

df = pd.read_csv(DATA_PATH)

print(df.head())

print("\nDataset Shape")

print(df.shape)

# =========================================================
# Feature Selection
# =========================================================

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

target = "likes"

X = df[features]

# Apply log transformation to likes
y = np.log1p(df[target])

print("\nFeatures Used")

print(features)

# =========================================================
# Train Test Split
# =========================================================

X_train, X_test, y_train, y_test = train_test_split(

    X,

    y,

    test_size=0.20,

    random_state=42

)

print("\nTraining Samples :", len(X_train))

print("Testing Samples :", len(X_test))

# =========================================================
# Models
# =========================================================

models = {

    "Linear Regression": LinearRegression(),

    "Ridge": Ridge(alpha=1.0),

    "Lasso": Lasso(alpha=0.1),

    "ElasticNet": ElasticNet(alpha=0.1,l1_ratio=0.5),

    "Random Forest": RandomForestRegressor(

        n_estimators=100,
        random_state=42,
        n_jobs=-1

    )

}

results = []

best_model = None

best_score = -999999

best_name = ""

print("\nStarting Training...\n")

# Training Loop
# =========================================================

for name, model in models.items():

    print("=" * 50)
    print(name)
    print("=" * 50)

    # Train model
    model.fit(X_train, y_train)

    # Predict (log scale)
    predictions_log = model.predict(X_test)

    # Convert predictions back to original likes
    predictions = np.expm1(predictions_log)

    # Convert actual values back to original likes
    y_test_original = np.expm1(y_test)

    mae = mean_absolute_error(
        y_test,
        predictions_log
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_test,
            predictions_log
        )
    )

    r2 = r2_score(
        y_test,
        predictions_log
    )

    # Cross Validation
    cv_score = cross_val_score(
        model,
        X_train,
        y_train,
        cv=3,
        scoring="r2",
        n_jobs=-1
    ).mean()

    print("MAE :", round(mae, 3))
    print("RMSE :", round(rmse, 3))
    print("R2 :", round(r2, 3))
    print("Cross Validation :", round(cv_score, 3))

    results.append({
        "Model": name,
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2,
        "Cross Validation": cv_score
    })

    filename = name.lower().replace(" ", "_") + ".pkl"

    joblib.dump(
        model,
        os.path.join(
            MODEL_DIR,
            filename
        )
    )

    if r2 > best_score:
        best_score = r2
        best_model = model
        best_name = name

print("\n" + "=" * 60)
print("Saving Best Model...")
print("=" * 60)

best_model_path = os.path.join(MODEL_DIR, "best_model.pkl")

joblib.dump(best_model, best_model_path)

print(f"Best Model: {best_name}")
print(f"Best R² Score: {best_score:.4f}")
print(f"Saved to: {best_model_path}")

# =========================================================
# Save Model Comparison Report
# =========================================================

results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    by="R2",
    ascending=False
)

report_path = os.path.join(
    REPORT_DIR,
    "model_results.csv"
)

results_df.to_csv(
    report_path,
    index=False
)

print("\nModel Comparison Report Saved")
print(report_path)

# =========================================================
# Display Results
# =========================================================

print("\n" + "=" * 80)
print("MODEL PERFORMANCE")
print("=" * 80)

print(results_df)

# =========================================================
# Feature Importance
# =========================================================

print("\n" + "=" * 60)
print("FEATURE IMPORTANCE")
print("=" * 60)

if hasattr(best_model, "feature_importances_"):

    importance = pd.DataFrame({

        "Feature": features,

        "Importance": best_model.feature_importances_

    })

    importance = importance.sort_values(
        by="Importance",
        ascending=False
    )

    print(importance)

    importance.to_csv(

        os.path.join(
            REPORT_DIR,
            "feature_importance.csv"
        ),

        index=False

    )

elif hasattr(best_model, "coef_"):

    importance = pd.DataFrame({

        "Feature": features,

        "Coefficient": best_model.coef_

    })

    importance = importance.sort_values(

        by="Coefficient",

        ascending=False

    )

    print(importance)

    importance.to_csv(

        os.path.join(
            REPORT_DIR,
            "feature_importance.csv"
        ),

        index=False

    )

else:

    print("Feature importance not available.")

# =========================================================
# Final Summary
# =========================================================

print("\n" + "=" * 80)
print("TRAINING COMPLETED SUCCESSFULLY")
print("=" * 80)

print(f"""
Total Models Trained : {len(models)}

Best Model           : {best_name}

Best R² Score        : {best_score:.4f}

Dataset Used         : featured_instagram.csv

Saved Models         : {MODEL_DIR}

Evaluation Report    : {report_path}
""")

print("=" * 80)