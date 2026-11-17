import pandas as pd

df = pd.read_csv("dataset/train.csv")

print(df["Credit_Mix"].unique())