import pandas as pd
import numpy as np

# ==============================
# 1. Load datasets
# ==============================

train = pd.read_csv("C:\\Users\\stefi\\credit_score_prjct\\dataset\\train.csv", low_memory=False)
test = pd.read_csv("C:\\Users\\stefi\\credit_score_prjct\\dataset\\test.csv", low_memory=False)

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

train = train.drop(columns=columns_to_drop)
test = test.drop(columns=columns_to_drop)


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

    train[column] = train[column].fillna(train[column].median())

    test[column] = test[column].fillna(train[column].median())


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
# 7. Display cleaned information
# ==============================

print("\nCleaned Train Shape:", train.shape)
print("Cleaned Test Shape:", test.shape)

print("\nMissing values after cleaning:")
print(train.isnull().sum())

print("\nData types after cleaning:")
print(train.dtypes)

print("\nCredit Score distribution:")
print(train["Credit_Score"].value_counts())


# ==============================
# 8. Save cleaned datasets
# ==============================

train.to_csv("dataset/train_cleaned.csv", index=False)
test.to_csv("dataset/test_cleaned.csv", index=False)

print("\nCleaned datasets saved successfully!")