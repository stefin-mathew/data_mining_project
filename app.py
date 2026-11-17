
from flask import Flask, render_template, request, jsonify
import joblib
import pandas as pd
import plotly.express as px
import plotly.io as pio
import os

app = Flask(__name__)

# ==============================
# LOAD TRAINED DECISION TREE MODEL
# ==============================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "credit_score_12_features.pkl"
)

model = joblib.load(MODEL_PATH)


# ==============================
# MODEL INPUT FEATURES
# ==============================

FEATURES = [
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

NUMERIC_FEATURES = [
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


# ==============================
# HOME PAGE
# ==============================

@app.route("/")
def home():
    return render_template("index.html")


# ==============================
# ANALYTICS PAGE
# ==============================

@app.route("/analytics")
def analytics():

    dataset_path = os.path.join(
        BASE_DIR,
        "dataset",
        "train_cleaned_final.csv"
    )

    df = pd.read_csv(dataset_path)

    # Credit Score Distribution

    score_counts = (
        df["Credit_Score"]
        .value_counts()
        .reset_index()
    )

    score_counts.columns = ["Credit_Score", "Count"]

    fig1 = px.bar(
        score_counts,
        x="Credit_Score",
        y="Count",
        title="Credit Score Distribution",
        text="Count",
        color="Credit_Score",
        color_discrete_map={
            "Good": "#16a34a",
            "Standard": "#2563eb",
            "Poor": "#dc2626"
        }
    )

    fig1.update_layout(
        plot_bgcolor="white",
        paper_bgcolor="white",
        font=dict(family="Arial", size=13),
        title_x=0.5
    )

    # Average Annual Income by Credit Score

    income_df = (
        df.groupby("Credit_Score")["Annual_Income"]
        .mean()
        .reset_index()
    )

    fig2 = px.bar(
        income_df,
        x="Credit_Score",
        y="Annual_Income",
        title="Average Annual Income by Credit Score",
        text_auto=".2s",
        color="Credit_Score",
        color_discrete_map={
            "Good": "#16a34a",
            "Standard": "#2563eb",
            "Poor": "#dc2626"
        }
    )

    fig2.update_layout(
        plot_bgcolor="white",
        paper_bgcolor="white",
        font=dict(family="Arial", size=13),
        title_x=0.5,
        yaxis_title="Average Annual Income"
    )

    # Convert charts into HTML

    charts = {
        "chart1": pio.to_html(fig1, full_html=False),
        "chart2": pio.to_html(fig2, full_html=False)
    }

    return render_template(
        "analytics.html",
        charts=charts
    )


# ==============================
# PREDICTION API
# ==============================

@app.route("/predict", methods=["POST"])
def predict():

    try:

        data = request.get_json(silent=True)

        if not isinstance(data, dict):
            return jsonify({
                "error": "Invalid JSON data received."
            }), 400

        # Check whether all 12 features are provided

        missing_features = [
            feature for feature in FEATURES
            if feature not in data
        ]

        if missing_features:
            return jsonify({
                "error": "Missing features: " +
                         ", ".join(missing_features)
            }), 400

        # Create input DataFrame in the correct feature order

        input_data = pd.DataFrame(
            [[data[feature] for feature in FEATURES]],
            columns=FEATURES
        )

        # Convert numerical inputs into numeric values

        for feature in NUMERIC_FEATURES:
            input_data[feature] = pd.to_numeric(
                input_data[feature],
                errors="coerce"
            )

        # Predict credit score

        prediction = model.predict(input_data)[0]

        return jsonify({
            "credit_score": str(prediction)
        })

    except Exception as e:

        return jsonify({
            "error": str(e)
        }), 500


# ==============================
# RUN FLASK APPLICATION
# ==============================

if __name__ == "__main__":
    app.run(debug=True)
