
import pandas as pd
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.impute import SimpleImputer

# Load the cleaned dataset
data = pd.read_csv("dataset/train_cleaned_final.csv")

# Features displayed in index.html
features = [
    "Age",
    "Annual_Income",
    "Outstanding_Debt",
    "Num_Bank_Accounts",
    "Num_Credit_Card",
    "Credit_Mix",
    "Delay_from_due_date",
    "Credit_Utilization_Ratio"
]

target = "Credit_Score"

# Check that the required columns exist
missing = [c for c in features + [target] if c not in data.columns]

if missing:
    raise ValueError(f"Missing columns in dataset: {missing}")

X = data[features].copy()
y = data[target].copy()

# Clean numeric and categorical inputs
numeric_features = [
    "Age",
    "Annual_Income",
    "Outstanding_Debt",
    "Num_Bank_Accounts",
    "Num_Credit_Card",
    "Delay_from_due_date",
    "Credit_Utilization_Ratio"
]

categorical_features = ["Credit_Mix"]

preprocessor = ColumnTransformer([
    ("numeric", SimpleImputer(strategy="median"), numeric_features),
    ("categorical", Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ]), categorical_features)
])

model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", DecisionTreeClassifier(
        criterion="entropy",
        random_state=42
    ))
])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model.fit(X_train, y_train)

print("Model accuracy:", model.score(X_test, y_test))

os.makedirs("models", exist_ok=True)

joblib.dump(model, "models/credit_score_8_features.pkl")

print("New model saved successfully!")
