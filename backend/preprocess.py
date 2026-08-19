import os
import pandas as pd
import pyarrow.parquet as pq
from sklearn.preprocessing import LabelEncoder

# =========================================================
# Paths
# =========================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_PATH = os.path.join(BASE_DIR, "dataset", "instagram.parquet")

OUTPUT_PATH = os.path.join(BASE_DIR, "dataset", "processed_instagram.csv")

# =========================================================
# Load Dataset
# =========================================================

print("=" * 60)
print("Loading Dataset...")
print("=" * 60)

try:
    table = pq.read_table(DATA_PATH)
    df = table.to_pandas()

    print("Dataset Loaded Successfully.\n")

except Exception as e:
    print("Error Loading Dataset")
    print(e)
    exit()

# =========================================================
# Basic Information
# =========================================================

print("=" * 60)
print("Dataset Overview")
print("=" * 60)

print("\nFirst Five Rows\n")
print(df.head())

print("\nShape :", df.shape)

print("\nColumn Names\n")
print(df.columns.tolist())

print("\nData Types\n")
print(df.dtypes)

print("\nDataset Information\n")
df.info()

print("\nMissing Values\n")
print(df.isnull().sum())

print("\nDuplicate Rows :", df.duplicated().sum())

print("\nSummary Statistics\n")
print(df.describe(include="all"))

# =========================================================
# Remove Unnecessary Columns
# =========================================================

print("\nRemoving Unnecessary Columns...")

drop_columns = [
    "sid",
    "profile_id",
    "shortcode",
    "username",
    "path"
]

existing_columns = [col for col in drop_columns if col in df.columns]

df.drop(columns=existing_columns, inplace=True)

print("Remaining Columns")

print(df.columns.tolist())

# =========================================================
# Remove Duplicate Rows
# =========================================================

print("\nRemoving Duplicate Rows...")

before = len(df)

df.drop_duplicates(inplace=True)

after = len(df)

print(f"Removed {before-after} duplicate rows.")

# =========================================================
# Handle Missing Values
# =========================================================

print("\nHandling Missing Values...")

# Numerical columns

num_cols = df.select_dtypes(include=["int64", "float64"]).columns

for col in num_cols:
    df[col] = df[col].fillna(df[col].median())

# Boolean columns

bool_cols = df.select_dtypes(include=["bool"]).columns

for col in bool_cols:
    df[col] = df[col].fillna(False)

# Object columns

cat_cols = df.select_dtypes(include=["object"]).columns

for col in cat_cols:

    if df[col].isnull().sum() > 0:

        df[col] = df[col].fillna(df[col].mode()[0])

print("\nMissing Values After Cleaning\n")

print(df.isnull().sum())

# =========================================================
# Encode Categorical Variables
# =========================================================

print("\nEncoding Categorical Variables...")

encoder = LabelEncoder()

categorical_columns = [

    "lang",

    "description_category"

]

for col in categorical_columns:

    if col in df.columns:

        df[col] = encoder.fit_transform(df[col].astype(str))

# Convert Boolean

if "is_business_account" in df.columns:

    df["is_business_account"] = df["is_business_account"].astype(int)

print("Encoding Completed.")

# =========================================================
# Final Dataset Information
# =========================================================

print("\nFinal Dataset Shape")

print(df.shape)

print("\nRemaining Missing Values")

print(df.isnull().sum().sum())

# =========================================================
# Save Dataset
# =========================================================

print("\nSaving Processed Dataset...")

os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

df.to_csv(OUTPUT_PATH, index=False)

print("Processed Dataset Saved Successfully")

print(OUTPUT_PATH)

print("\nPreprocessing Completed Successfully!")