
import pandas as pd
import numpy as np
import os

# ==============================
# 1. Load datasets
# ==============================

BASE_DIR = r"C:\Users\stefi\credit_score_prjct"

train = pd.read_csv(
    os.path.join(BASE_DIR, "dataset", "train.csv"),
    low_memory=False
)

test = pd.read_csv(
    os.path.join(BASE_DIR, "dataset", "test.csv"),
    low_memory=False
)

print("Original train shape:", train.shape)
print("Original test shape:", test.shape)


# ==============================
# 2. Remove unnecessary columns
# ==============================

columns_to_drop = [
    "ID",
    "Customer_ID",
    "Name",
    "SSN"
]

train.drop(
    columns=columns_to_drop,
    errors="ignore",
    inplace=True
)

test.drop(
    columns=columns_to_drop,
    errors="ignore",
    inplace=True
)


# ==============================
# 3. Clean numerical columns
# ==============================

numeric_columns = [
    "Age",
    "Annual_Income",
    "Monthly_Inhand_Salary",
    "Num_Bank_Accounts",
    "Num_Credit_Card",
    "Interest_Rate",
    "Num_of_Loan",
    "Delay_from_due_date",
    "Num_of_Delayed_Payment",
    "Changed_Credit_Limit",
    "Num_Credit_Inquiries",
    "Outstanding_Debt",
    "Credit_Utilization_Ratio",
    "Total_EMI_per_month",
    "Amount_invested_monthly",
    "Monthly_Balance"
]

for column in numeric_columns:

    train[column] = pd.to_numeric(
        train[column].astype(str).str.replace("_", "", regex=False),
        errors="coerce"
    )

    test[column] = pd.to_numeric(
        test[column].astype(str).str.replace("_", "", regex=False),
        errors="coerce"
    )


# ==============================
# 4. Handle missing numerical values
# ==============================

for column in numeric_columns:

    median_value = train[column].median()

    train[column] = train[column].fillna(median_value)

    test[column] = test[column].fillna(median_value)


# ==============================
# 5. Handle missing categorical values
# ==============================

categorical_columns = [
    "Occupation",
    "Type_of_Loan",
    "Credit_Mix",
    "Payment_of_Min_Amount",
    "Payment_Behaviour"
]

for column in categorical_columns:

    train[column] = train[column].fillna("Unknown")
    test[column] = test[column].fillna("Unknown")


# ==============================
# 6. Clean categorical values
# ==============================

for column in categorical_columns:

    train[column] = train[column].astype(str).str.strip()
    test[column] = test[column].astype(str).str.strip()


# ==============================
# 7. Select 12 model features
# ==============================

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

# Check whether required columns exist

missing_train = [
    col for col in features + [target]
    if col not in train.columns
]

missing_test = [
    col for col in features
    if col not in test.columns
]

if missing_train or missing_test:
    raise ValueError(
        f"Missing train columns: {missing_train}\n"
        f"Missing test columns: {missing_test}"
    )

train_final = train[features + [target]].copy()
test_final = test[features].copy()


# ==============================
# 8. Display cleaned information
# ==============================

print("\nCleaned Train Shape:", train_final.shape)
print("Cleaned Test Shape:", test_final.shape)

print("\nMissing values in training data:")
print(train_final.isnull().sum())

print("\nData types:")
print(train_final.dtypes)

print("\nCredit Score Distribution:")
print(train_final["Credit_Score"].value_counts())


# ==============================
# 9. Save cleaned datasets
# ==============================

dataset_folder = os.path.join(BASE_DIR, "dataset")

train_final.to_csv(
    os.path.join(dataset_folder, "train_cleaned_final.csv"),
    index=False
)

test_final.to_csv(
    os.path.join(dataset_folder, "test_cleaned_final.csv"),
    index=False
)

print("\nFinal cleaned datasets saved successfully!")
