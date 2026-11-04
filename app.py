from flask import Flask, render_template, request, jsonify
import joblib
import pandas as pd
import plotly.express as px
import plotly.io as pio

app = Flask(__name__)

# Load trained Random Forest model
model = joblib.load("models/credit_score_8_features.pkl")

# Home page
@app.route("/")
def home():
    return render_template("index.html")


# Analytics page
@app.route("/analytics")
def analytics():

    df = pd.read_csv("dataset/train_cleaned_final.csv")

    # Credit Score Distribution
    score_counts = df["Credit_Score"].value_counts().reset_index()
    score_counts.columns = ["Credit_Score", "Count"]

    fig1 = px.bar(
        score_counts,
        x="Credit_Score",
        y="Count",
        title="Credit Score Distribution",
        text="Count"
    )

    # Average Annual Income
    income_df = (
        df.groupby("Credit_Score")["Annual_Income"]
        .mean()
        .reset_index()
    )

    fig2 = px.bar(
        income_df,
        x="Credit_Score",
        y="Annual_Income",
        title="Average Annual Income by Credit Score"
    )

    # Convert charts to HTML
    charts = {
        "chart1": pio.to_html(fig1, full_html=False),
        "chart2": pio.to_html(fig2, full_html=False)
    }

    return render_template(
        "analytics.html",
        charts=charts
    )


# Prediction API
@app.route("/predict", methods=["POST"])
def predict():

    data = request.get_json()

    input_data = pd.DataFrame([data])

    prediction = model.predict(input_data)[0]

    return jsonify({
        "credit_score": prediction
    })


# Start Flask
if __name__ == "__main__":
    app.run(debug=True)