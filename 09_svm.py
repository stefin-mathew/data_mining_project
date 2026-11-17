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

# ==========================================
# 11. SVM MARGIN AND SUPPORT VECTORS
# ==========================================

import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC

# Create sample 2D data for margin illustration
np.random.seed(42)

X_visual = np.array([
    [2, 3], [3, 3], [3, 4], [4, 3],
    [6, 7], [7, 7], [7, 8], [8, 7]
])

y_visual = np.array([0, 0, 0, 0, 1, 1, 1, 1])

# Train linear SVM for visualization
svm_visual = SVC(kernel="linear", C=1000)

svm_visual.fit(X_visual, y_visual)

# Get hyperplane parameters
w = svm_visual.coef_[0]
b = svm_visual.intercept_[0]

# Calculate margin
margin = 1 / np.sqrt(np.sum(w ** 2))

# Create hyperplane
x_axis = np.linspace(0, 10, 100)

y_hyperplane = -(w[0] * x_axis + b) / w[1]

y_margin_positive = (
    -(w[0] * x_axis + b - 1) / w[1]
)

y_margin_negative = (
    -(w[0] * x_axis + b + 1) / w[1]
)

# Plot
plt.figure(figsize=(9, 7))

# Data points
plt.scatter(
    X_visual[y_visual == 0, 0],
    X_visual[y_visual == 0, 1],
    label="Class 0",
    s=80
)

plt.scatter(
    X_visual[y_visual == 1, 0],
    X_visual[y_visual == 1, 1],
    label="Class 1",
    s=80
)

# Hyperplane
plt.plot(
    x_axis,
    y_hyperplane,
    "k-",
    label="Hyperplane"
)

# Margins
plt.plot(
    x_axis,
    y_margin_positive,
    "r--",
    label="Positive Margin"
)

plt.plot(
    x_axis,
    y_margin_negative,
    "b--",
    label="Negative Margin"
)

# Highlight support vectors
plt.scatter(
    svm_visual.support_vectors_[:, 0],
    svm_visual.support_vectors_[:, 1],
    s=200,
    facecolors="none",
    edgecolors="black",
    linewidths=2,
    label="Support Vectors"
)

plt.fill_between(
    x_axis,
    y_margin_negative,
    y_margin_positive,
    alpha=0.15
)

plt.title("SVM Maximum Margin and Support Vectors")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")

plt.legend()
plt.grid(True)
plt.xlim(0, 10)
plt.ylim(0, 10)

plt.tight_layout()
plt.show()

print("Margin width:", 2 * margin)
print("Support Vectors:", len(svm_visual.support_vectors_))
