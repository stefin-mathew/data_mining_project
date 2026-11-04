import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

# ==========================================
# 1. Load cleaned dataset
# ==========================================

df = pd.read_csv("dataset/train_cleaned_final.csv")

# ==========================================
# 2. Separate features and target
# ==========================================

X = df.drop("Credit_Score", axis=1)
y = df["Credit_Score"]

# ==========================================
# 3. Identify columns
# ==========================================

categorical_columns = X.select_dtypes(
    include=["object", "str"]
).columns.tolist()

numerical_columns = X.select_dtypes(
    include=["int64", "float64"]
).columns.tolist()

# ==========================================
# 4. Preprocessing
# ==========================================

preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_columns
        )
    ],
    remainder="passthrough"
)

# ==========================================
# 5. Random Forest model
# ==========================================

model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
           ("classifier", RandomForestClassifier(
            n_estimators=50,
            max_depth=15,
            min_samples_leaf=2,
            random_state=42,
            class_weight="balanced",
            n_jobs=-1
        ))
        )
    ]
)

# ==========================================
# 6. Train using complete training dataset
# ==========================================

print("Training final Random Forest model...")

model.fit(X, y)

print("Training completed!")

# ==========================================
# 7. Save model
# ==========================================

joblib.dump(
    model,
    "models/random_forest_credit_score.pkl",
    compress=3
)

print("\nModel saved successfully!")
print("Location: models/random_forest_credit_score.pkl")