import pandas as pd

train = pd.read_csv("C:\\Users\\stefi\\credit_score_prjct\\dataset\\train.csv")
test = pd.read_csv("C:\\Users\\stefi\\credit_score_prjct\\dataset\\test.csv")

print("Train shape:", train.shape)
print("Test shape:", test.shape)

print("\nFirst 5 rows:")
print(train.head())

print("\nColumn names:")
print(train.columns)

print("\nData types:")
print(train.dtypes)

print("\nMissing values:")
print(train.isnull().sum())

print("\nTarget distribution:")
print(train["Credit_Score"].value_counts())