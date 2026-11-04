import pandas as pd
import numpy as np

# Load cleaned dataset
train = pd.read_csv("dataset/train_cleaned.csv")
test = pd.read_csv("dataset/test_cleaned.csv")

# Function to convert "X Years and Y Months" into total months
def convert_credit_history(value):
    if pd.isna(value):
        return np.nan

    value = str(value)

    try:
        years = int(value.split(" Years")[0])
        months = int(value.split("and ")[1].split(" Months")[0])

        return years * 12 + months

    except:
        return np.nan


# Convert Credit_History_Age into months
train["Credit_History_Age"] = train["Credit_History_Age"].apply(
    convert_credit_history
)

test["Credit_History_Age"] = test["Credit_History_Age"].apply(
    convert_credit_history
)

# Fill missing values using training median
median_value = train["Credit_History_Age"].median()

train["Credit_History_Age"] = train["Credit_History_Age"].fillna(
    median_value
)

test["Credit_History_Age"] = test["Credit_History_Age"].fillna(
    median_value
)

# Check result
print("Missing Credit_History_Age in train:",
      train["Credit_History_Age"].isnull().sum())

print("Missing Credit_History_Age in test:",
      test["Credit_History_Age"].isnull().sum())

print("\nData type:")
print(train["Credit_History_Age"].dtype)

# Save final cleaned datasets
train.to_csv("dataset/train_cleaned_final.csv", index=False)
test.to_csv("dataset/test_cleaned_final.csv", index=False)

print("\nFinal cleaned datasets saved successfully!")