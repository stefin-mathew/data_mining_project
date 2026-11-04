import pandas as pd
import joblib

# Load saved model
model = joblib.load(
    "models/random_forest_credit_score.pkl"
)

# Load test dataset
test = pd.read_csv(
    "dataset/test_cleaned_final.csv"
)

# Select first 5 customers
sample = test.head(5)

# Make predictions
predictions = model.predict(sample)

print("Predicted Credit Scores:")

for i, prediction in enumerate(predictions):
    print(f"Customer {i + 1}: {prediction}")