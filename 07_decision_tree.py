
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# 1. Load data
df = pd.read_csv("dataset/train_cleaned_final.csv")

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

X = df[features]
y = df["Credit_Score"]

# 3. Identify columns
categorical_columns = X.select_dtypes(
    exclude=["number"]
).columns.tolist()

numerical_columns = X.select_dtypes(
    include=["number"]
).columns.tolist()

# 4. Preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", OneHotEncoder(handle_unknown="ignore"))
            ]),
            categorical_columns
        ),
        (
            "numerical",
            SimpleImputer(strategy="median"),
            numerical_columns
        )
    ]
)

# 5. Split data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# 6. Decision Tree model
model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "classifier",
            DecisionTreeClassifier(
                criterion="entropy",
                random_state=42
            )
        )
    ]
)

# 7. Train
print("Training Decision Tree...")

model.fit(X_train, y_train)

print("Training completed!")

# 8. Prediction
y_pred = model.predict(X_test)

# 9. Evaluation
accuracy = accuracy_score(y_test, y_pred)

print("\n==============================")
print("DECISION TREE RESULTS")
print("==============================")

print("\nAccuracy:", accuracy)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

cm = confusion_matrix(
    y_test,
    y_pred,
    labels=model.classes_
)

print("\nConfusion Matrix:")
print(cm)


# =========================================
# FIGURE 1: CREDIT SCORE DISTRIBUTION
# =========================================

plt.figure(figsize=(7, 5))

sns.countplot(
    data=df,
    x="Credit_Score",
    order=["Good", "Standard", "Poor"]
)

plt.title("Credit Score Distribution")
plt.xlabel("Credit Score")
plt.ylabel("Number of Customers")
plt.tight_layout()
plt.show()


# =========================================
# FIGURE 2: CONFUSION MATRIX
# =========================================

plt.figure(figsize=(7, 5))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=model.classes_,
    yticklabels=model.classes_
)

plt.title("Decision Tree - Confusion Matrix")
plt.xlabel("Predicted Credit Score")
plt.ylabel("Actual Credit Score")
plt.tight_layout()
plt.show()


# =========================================
# FIGURE 3: CLASSIFICATION REPORT
# =========================================

report = classification_report(
    y_test,
    y_pred,
    output_dict=True
)

report_df = pd.DataFrame(report).transpose()

report_df = report_df.loc[
    ["Good", "Poor", "Standard"],
    ["precision", "recall", "f1-score"]
]

report_df.plot(
    kind="bar",
    figsize=(9, 5),
    ylim=(0, 1),
    rot=0
)

plt.title("Decision Tree - Classification Report")
plt.xlabel("Credit Score")
plt.ylabel("Score")
plt.legend(title="Metrics")
plt.tight_layout()
plt.show()


# =========================================
# FIGURE 4: DECISION TREE VISUALIZATION
# =========================================

trained_preprocessor = model.named_steps["preprocessor"]
trained_tree = model.named_steps["classifier"]

feature_names = trained_preprocessor.get_feature_names_out()

plt.figure(figsize=(24, 12))

plot_tree(
    trained_tree,
    feature_names=feature_names,
    class_names=[str(c) for c in trained_tree.classes_],
    filled=True,
    rounded=True,
    max_depth=3,
    fontsize=8
)

plt.title("Decision Tree Visualization (First 3 Levels)")
plt.tight_layout()
plt.show()


# =========================================
# FINAL ACCURACY
# =========================================

print("\nFinal Decision Tree Accuracy:",
      round(accuracy * 100, 2), "%")

