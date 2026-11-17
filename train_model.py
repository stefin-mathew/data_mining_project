
import pandas as pd
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.impute import SimpleImputer

# 1. Load the cleaned dataset
data = pd.read_csv("dataset/train_cleaned_final.csv")

# 2. Select 12 features
features = [
    "Age",
    "Annual_Income",
    "Monthly_Inhand_Salary",
    "Outstanding_Debt",
    "Num_Bank_Accounts",
    "Num_Credit_Card",
    "Num_of_Loan",
    "Interest_Rate",
    "Num_of_Delayed_Payment",
    "Credit_Mix",
    "Delay_from_due_date",
    "Credit_Utilization_Ratio"
]

target = "Credit_Score"

# 3. Check required columns
missing = [c for c in features + [target] if c not in data.columns]

if missing:
    raise ValueError(f"Missing columns in dataset: {missing}")

# 4. Separate features and target
X = data[features].copy()
y = data[target].copy()

# 5. Identify numerical and categorical columns
numeric_features = X.select_dtypes(
    include=["number"]
).columns.tolist()

categorical_features = X.select_dtypes(
    exclude=["number"]
).columns.tolist()

# 6. Preprocessing
preprocessor = ColumnTransformer([
    (
        "numeric",
        SimpleImputer(strategy="median"),
        numeric_features
    ),
    (
        "categorical",
        Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore"))
        ]),
        categorical_features
    )
])

# 7. Decision Tree model
model = Pipeline([
    ("preprocessor", preprocessor),
    (
        "classifier",
        DecisionTreeClassifier(
            criterion="entropy",
            random_state=42
        )
    )
])

# 8. Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# 9. Train model
model.fit(X_train, y_train)

# 10. Evaluate model
print("Model accuracy:", model.score(X_test, y_test))

# 11. Save model
os.makedirs("models", exist_ok=True)

joblib.dump(
    model,
    "models/credit_score_12_features.pkl"
)

print("12-feature Decision Tree model saved successfully!")
