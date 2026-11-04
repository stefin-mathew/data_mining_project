import pandas as pd
import matplotlib.pyplot as plt

# Load final cleaned training data
train = pd.read_csv("dataset/train_cleaned_final.csv")

# ==============================
# 1. Basic information
# ==============================

print("Dataset shape:", train.shape)

print("\nCredit Score Distribution:")
print(train["Credit_Score"].value_counts())

# ==============================
# 2. Credit Score Distribution
# ==============================

plt.figure(figsize=(7, 5))

train["Credit_Score"].value_counts().plot(kind="bar")

plt.title("Credit Score Distribution")
plt.xlabel("Credit Score")
plt.ylabel("Number of Customers")
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()
# ==============================
# 7. Numerical Analysis
# ==============================

print("\nAverage values by Credit Score:")

summary = train.groupby("Credit_Score")[
    [
        "Annual_Income",
        "Outstanding_Debt",
        "Interest_Rate",
        "Num_of_Delayed_Payment",
        "Credit_Utilization_Ratio"
    ]
].mean()

print(summary)
# ==============================
# 8. Correlation Analysis
# ==============================

numeric_data = train.select_dtypes(include=["int64", "float64"])

correlation = numeric_data.corr()

print("\nCorrelation with Credit-related variables:")
print(correlation.round(2))