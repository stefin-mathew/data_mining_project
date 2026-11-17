
import pandas as pd
import matplotlib.pyplot as plt

# ==============================
# 1. Load cleaned training data
# ==============================

train = pd.read_csv("dataset/train_cleaned_final.csv")

print("Dataset Shape:", train.shape)

print("\nDataset Information:")
print(train.info())

# ==============================
# 2. Credit Score Distribution
# ==============================

print("\nCredit Score Distribution:")
print(train["Credit_Score"].value_counts())

plt.figure(figsize=(7, 5))

train["Credit_Score"].value_counts().plot(
    kind="bar",
    color=["skyblue", "orange", "lightgreen"]
)

plt.title("Credit Score Distribution")
plt.xlabel("Credit Score")
plt.ylabel("Number of Customers")
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()


# ==============================
# 3. Credit Mix Distribution
# ==============================

print("\nCredit Mix Distribution:")
print(train["Credit_Mix"].value_counts())

plt.figure(figsize=(7, 5))

train["Credit_Mix"].value_counts().plot(
    kind="bar",
    color="mediumpurple"
)

plt.title("Credit Mix Distribution")
plt.xlabel("Credit Mix")
plt.ylabel("Number of Customers")
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()


# ==============================
# 4. Numerical Analysis
# ==============================

numeric_features = [
    "Age",
    "Annual_Income",
    "Monthly_Inhand_Salary",
    "Outstanding_Debt",
    "Num_Bank_Accounts",
    "Num_Credit_Card",
    "Num_of_Loan",
    "Interest_Rate",
    "Num_of_Delayed_Payment",
    "Delay_from_due_date",
    "Credit_Utilization_Ratio"
]

print("\nAverage Numerical Values by Credit Score:")

summary = train.groupby("Credit_Score")[numeric_features].mean()

print(summary.round(2))


# ==============================
# 5. Feature Distribution
# ==============================

for feature in numeric_features:

    plt.figure(figsize=(7, 5))

    train[feature].hist(
        bins=30,
        edgecolor="black"
    )

    plt.title("Distribution of " + feature)
    plt.xlabel(feature)
    plt.ylabel("Frequency")

    plt.tight_layout()
    plt.show()


# ==============================
# 6. Average Values by Credit Score
# ==============================

selected_features = [
    "Annual_Income",
    "Outstanding_Debt",
    "Interest_Rate",
    "Num_of_Delayed_Payment",
    "Credit_Utilization_Ratio"
]

for feature in selected_features:

    average = train.groupby("Credit_Score")[feature].mean()

    plt.figure(figsize=(7, 5))

    average.plot(
        kind="bar",
        color="cornflowerblue"
    )

    plt.title("Average " + feature + " by Credit Score")
    plt.xlabel("Credit Score")
    plt.ylabel("Average " + feature)
    plt.xticks(rotation=0)

    plt.tight_layout()
    plt.show()


# ==============================
# 7. Correlation Analysis
# ==============================

numeric_data = train[numeric_features]

correlation = numeric_data.corr()

print("\nCorrelation Matrix:")
print(correlation.round(2))


# ==============================
# 8. Correlation Heatmap
# ==============================

plt.figure(figsize=(12, 9))

plt.imshow(
    correlation,
    cmap="coolwarm",
    interpolation="nearest",
    aspect="auto",
    vmin=-1,
    vmax=1
)

plt.colorbar(label="Correlation")

plt.xticks(
    range(len(numeric_features)),
    numeric_features,
    rotation=90
)

plt.yticks(
    range(len(numeric_features)),
    numeric_features
)

plt.title("Correlation Heatmap of Numerical Features")

plt.tight_layout()
plt.show()


# ==============================
# 9. Final EDA Summary
# ==============================

print("\n========== EDA SUMMARY ==========")

print("Total Records:", len(train))
print("Total Features:", len(numeric_features) + 1)
print("Numerical Features:", len(numeric_features))
print("Categorical Features: 1")
print("Target Variable: Credit_Score")

print("\nEDA Completed Successfully!")
