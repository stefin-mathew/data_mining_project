import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# ==========================================
# 1. Load data
# ==========================================

df = pd.read_csv("dataset/train_cleaned_final.csv")

# ==========================================
# 2. Use a stratified sample for SVM
# ==========================================

df_sample, _ = train_test_split(
    df,
    train_size=30000,
    random_state=42,
    stratify=df["Credit_Score"]
)

print("SVM dataset:", df_sample.shape)

# ==========================================
# 3. Features and target
# ==========================================

X = df_sample.drop("Credit_Score", axis=1)
y = df_sample["Credit_Score"]

# ==========================================
# 4. Identify columns
# ==========================================

categorical_columns = X.select_dtypes(
    include=["object", "str"]
).columns.tolist()

numerical_columns = X.select_dtypes(
    include=["int64", "float64"]
).columns.tolist()

# ==========================================
# 5. Preprocessing
# ==========================================

preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_columns
        ),
        (
            "numerical",
            StandardScaler(),
            numerical_columns
        )
    ]
)

# ==========================================
# 6. Train-Test Split
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# ==========================================
# 7. SVM Model
# ==========================================

model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "classifier",
            SVC(
                kernel="rbf",
                class_weight="balanced"
            )
        )
    ]
)

# ==========================================
# 8. Train
# ==========================================

print("\nTraining SVM...")
print("This may take some time...")

model.fit(X_train, y_train)

print("Training completed!")

# ==========================================
# 9. Prediction
# ==========================================

y_pred = model.predict(X_test)

# ==========================================
# 10. Evaluation
# ==========================================

accuracy = accuracy_score(y_test, y_pred)

print("\n==============================")
print("SVM RESULTS")
print("==============================")

print("\nAccuracy:", accuracy)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))