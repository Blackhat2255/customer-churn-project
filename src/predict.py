# ============================================================
# Customer Churn Prediction & Business Intelligence System
# Module: Customer Risk Scoring and Prediction
# ============================================================

import os
import warnings

import joblib
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


# ------------------------------------------------------------
# Configuration
# ------------------------------------------------------------

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

DATA_PATH = os.path.join(
    BASE_DIR,
    "data",
    "processed",
    "featured_churn_data.csv"
)

MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "best_churn_model.pkl"
)

REPORT_DIR = os.path.join(
    BASE_DIR,
    "reports"
)

MODEL_THRESHOLD = 0.30
HIGH_RISK_THRESHOLD = 0.70

os.makedirs(REPORT_DIR, exist_ok=True)


# ------------------------------------------------------------
# Display Helper
# ------------------------------------------------------------

def print_section(title):

    print()
    print("=" * 70)
    print(title)
    print("=" * 70)


# ------------------------------------------------------------
# Retention Recommendation
# ------------------------------------------------------------

def generate_recommendation(row):

    recommendations = []

    if row["Contract"] == "Month-to-month":
        recommendations.append(
            "Offer a longer-term contract with an incentive."
        )

    if row["tenure"] <= 12:
        recommendations.append(
            "Provide an early-tenure retention offer or onboarding support."
        )

    if row["MonthlyCharges"] >= 80:
        recommendations.append(
            "Review pricing and offer a suitable plan or discount."
        )

    if row["TechSupport"] == "No":
        recommendations.append(
            "Offer technical support or a support package."
        )

    if row["OnlineSecurity"] == "No":
        recommendations.append(
            "Recommend an online security service."
        )

    if row["InternetService"] == "Fiber optic":
        recommendations.append(
            "Review fiber service experience and address possible service concerns."
        )

    if row["PaymentMethod"] == "Electronic check":
        recommendations.append(
            "Consider encouraging automatic payment options."
        )

    if not recommendations:
        recommendations.append(
            "Monitor customer behavior and maintain service engagement."
        )

    return " ".join(recommendations[:3])


# ------------------------------------------------------------
# Load Data
# ------------------------------------------------------------

print_section("CUSTOMER CHURN RISK SCORING")

print("Loading feature-engineered dataset...")

df = pd.read_csv(DATA_PATH)

print("Dataset loaded successfully.")
print(f"Rows: {df.shape[0]}")
print(f"Columns: {df.shape[1]}")


# ------------------------------------------------------------
# Load Model
# ------------------------------------------------------------

print()
print("Loading trained churn model...")

model = joblib.load(MODEL_PATH)

print("Model loaded successfully.")
print("Model: Gradient Boosting")


# ------------------------------------------------------------
# Prepare Prediction Features
# ------------------------------------------------------------

X = df.drop(
    columns=["Churn", "customerID"]
)


# ------------------------------------------------------------
# Generate Churn Probabilities
# ------------------------------------------------------------

print_section("GENERATING CUSTOMER CHURN PROBABILITIES")

churn_probability = model.predict_proba(X)[:, 1]

df["Churn_Probability"] = churn_probability


# ------------------------------------------------------------
# Apply Business Threshold
# ------------------------------------------------------------

df["Predicted_Churn"] = (
    df["Churn_Probability"] >= MODEL_THRESHOLD
).astype(int)


# ------------------------------------------------------------
# Assign Risk Levels
# ------------------------------------------------------------

def assign_risk_level(probability):

    if probability >= HIGH_RISK_THRESHOLD:
        return "HIGH"

    elif probability >= MODEL_THRESHOLD:
        return "MEDIUM"

    else:
        return "LOW"


df["Risk_Level"] = df["Churn_Probability"].apply(
    assign_risk_level
)


# ------------------------------------------------------------
# Generate Retention Recommendations
# ------------------------------------------------------------

print()
print("Generating retention recommendations...")

df["Retention_Recommendation"] = df.apply(
    generate_recommendation,
    axis=1
)


# ------------------------------------------------------------
# Customer Risk Summary
# ------------------------------------------------------------

print_section("CUSTOMER RISK SUMMARY")

total_customers = len(df)

predicted_churn = (
    df["Predicted_Churn"].sum()
)

high_risk_customers = (
    df["Risk_Level"] == "HIGH"
).sum()

medium_risk_customers = (
    df["Risk_Level"] == "MEDIUM"
).sum()

low_risk_customers = (
    df["Risk_Level"] == "LOW"
).sum()


print(f"Total customers       : {total_customers:,}")
print(f"Predicted churn       : {predicted_churn:,}")
print(f"High-risk customers   : {high_risk_customers:,}")
print(f"Medium-risk customers : {medium_risk_customers:,}")
print(f"Low-risk customers    : {low_risk_customers:,}")


print()
print("Risk distribution:")

risk_distribution = (
    df["Risk_Level"]
    .value_counts()
    .reindex(["HIGH", "MEDIUM", "LOW"])
    .fillna(0)
    .astype(int)
)

print(risk_distribution)


# ------------------------------------------------------------
# Risk Percentages
# ------------------------------------------------------------

risk_percentage = (
    risk_distribution / total_customers * 100
).round(2)

print()
print("Risk distribution percentage:")

for level in ["HIGH", "MEDIUM", "LOW"]:

    print(
        f"{level}: "
        f"{risk_percentage[level]:.2f}%"
    )


# ------------------------------------------------------------
# Revenue Risk Analysis
# ------------------------------------------------------------

print_section("REVENUE AT RISK ANALYSIS")

high_risk = df[
    df["Risk_Level"] == "HIGH"
].copy()

medium_risk = df[
    df["Risk_Level"] == "MEDIUM"
].copy()

# Current monthly revenue from high-risk customers
high_risk_monthly_revenue = (
    high_risk["MonthlyCharges"].sum()
)

# Annualized revenue from high-risk customers
high_risk_annual_revenue = (
    high_risk_monthly_revenue * 12
)

# Probability-weighted revenue risk
probability_weighted_monthly_risk = (
    df["MonthlyCharges"] *
    df["Churn_Probability"]
).sum()

probability_weighted_annual_risk = (
    probability_weighted_monthly_risk * 12
)


print(
    f"High-risk monthly revenue: "
    f"${high_risk_monthly_revenue:,.2f}"
)

print(
    f"High-risk annual revenue: "
    f"${high_risk_annual_revenue:,.2f}"
)

print(
    f"Probability-weighted monthly risk: "
    f"${probability_weighted_monthly_risk:,.2f}"
)

print(
    f"Probability-weighted annual risk: "
    f"${probability_weighted_annual_risk:,.2f}"
)


# ------------------------------------------------------------
# Average Risk Metrics
# ------------------------------------------------------------

average_churn_probability = (
    df["Churn_Probability"].mean()
)

average_high_risk_probability = (
    high_risk["Churn_Probability"].mean()
    if len(high_risk) > 0
    else 0
)

print()
print(
    f"Average churn probability: "
    f"{average_churn_probability:.4f}"
)

print(
    f"Average HIGH-risk probability: "
    f"{average_high_risk_probability:.4f}"
)


# ------------------------------------------------------------
# Top High-Risk Customers
# ------------------------------------------------------------

print_section("TOP HIGH-RISK CUSTOMERS")

display_columns = [
    "customerID",
    "Contract",
    "tenure",
    "InternetService",
    "TechSupport",
    "OnlineSecurity",
    "PaymentMethod",
    "MonthlyCharges",
    "Churn_Probability",
    "Risk_Level",
    "Retention_Recommendation"
]

top_high_risk = (
    high_risk[
        display_columns
    ]
    .sort_values(
        by="Churn_Probability",
        ascending=False
    )
    .head(20)
)

if len(top_high_risk) > 0:

    output_display = top_high_risk.copy()

    output_display["Churn_Probability"] = (
        output_display["Churn_Probability"] * 100
    ).round(2)

    print(
        output_display.to_string(
            index=False
        )
    )

else:

    print("No HIGH-risk customers found.")


# ------------------------------------------------------------
# Segment Risk Analysis
# ------------------------------------------------------------

print_section("RISK BY CUSTOMER SEGMENT")

segment_summary = (
    df.groupby("Customer_Segment")
    .agg(
        Customers=("customerID", "count"),
        Average_Churn_Probability=(
            "Churn_Probability",
            "mean"
        ),
        High_Risk_Customers=(
            "Risk_Level",
            lambda x: (x == "HIGH").sum()
        ),
        Monthly_Revenue=(
            "MonthlyCharges",
            "sum"
        )
    )
    .reset_index()
)

segment_summary[
    "Average_Churn_Probability"
] *= 100

segment_summary[
    "Average_Churn_Probability"
] = segment_summary[
    "Average_Churn_Probability"
].round(2)

print(
    segment_summary.to_string(
        index=False
    )
)


# ------------------------------------------------------------
# Contract Risk Analysis
# ------------------------------------------------------------

print_section("RISK BY CONTRACT TYPE")

contract_summary = (
    df.groupby("Contract")
    .agg(
        Customers=("customerID", "count"),
        Average_Churn_Probability=(
            "Churn_Probability",
            "mean"
        ),
        High_Risk_Customers=(
            "Risk_Level",
            lambda x: (x == "HIGH").sum()
        ),
        Monthly_Revenue=(
            "MonthlyCharges",
            "sum"
        )
    )
    .reset_index()
)

contract_summary[
    "Average_Churn_Probability"
] *= 100

contract_summary[
    "Average_Churn_Probability"
] = contract_summary[
    "Average_Churn_Probability"
].round(2)

print(
    contract_summary.to_string(
        index=False
    )
)


# ------------------------------------------------------------
# Save Complete Risk Dataset
# ------------------------------------------------------------

risk_output_path = os.path.join(
    REPORT_DIR,
    "customer_risk_scores.csv"
)

df.to_csv(
    risk_output_path,
    index=False
)

print()
print("Complete customer risk dataset saved to:")
print(risk_output_path)


# ------------------------------------------------------------
# Save High-Risk Customers
# ------------------------------------------------------------

high_risk_output_path = os.path.join(
    REPORT_DIR,
    "high_risk_customers.csv"
)

high_risk.to_csv(
    high_risk_output_path,
    index=False
)

print()
print("High-risk customer list saved to:")
print(high_risk_output_path)


# ------------------------------------------------------------
# Save Risk Summary
# ------------------------------------------------------------

summary_data = {
    "Metric": [
        "Total Customers",
        "Predicted Churn Customers",
        "High-Risk Customers",
        "Medium-Risk Customers",
        "Low-Risk Customers",
        "Overall Average Churn Probability",
        "High-Risk Monthly Revenue",
        "High-Risk Annual Revenue",
        "Probability-Weighted Monthly Revenue Risk",
        "Probability-Weighted Annual Revenue Risk",
        "Final Business Threshold"
    ],

    "Value": [
        total_customers,
        predicted_churn,
        high_risk_customers,
        medium_risk_customers,
        low_risk_customers,
        average_churn_probability,
        high_risk_monthly_revenue,
        high_risk_annual_revenue,
        probability_weighted_monthly_risk,
        probability_weighted_annual_risk,
        MODEL_THRESHOLD
    ]
}

risk_summary = pd.DataFrame(
    summary_data
)

risk_summary_path = os.path.join(
    REPORT_DIR,
    "risk_summary.csv"
)

risk_summary.to_csv(
    risk_summary_path,
    index=False
)


# ------------------------------------------------------------
# Manual Customer Prediction Function
# ------------------------------------------------------------

def predict_customer(customer_id):

    customer = df[
        df["customerID"] == customer_id
    ]

    if customer.empty:

        print()
        print(
            f"Customer ID '{customer_id}' "
            f"was not found."
        )

        return

    customer_features = customer.drop(
        columns=[
            "Churn",
            "customerID",
            "Churn_Probability",
            "Predicted_Churn",
            "Risk_Level",
            "Retention_Recommendation"
        ]
    )

    probability = model.predict_proba(
        customer_features
    )[:, 1][0]

    prediction = (
        probability >= MODEL_THRESHOLD
    )

    if probability >= HIGH_RISK_THRESHOLD:

        risk = "HIGH"

    elif probability >= MODEL_THRESHOLD:

        risk = "MEDIUM"

    else:

        risk = "LOW"

    print()
    print_section("INDIVIDUAL CUSTOMER PREDICTION")

    print(
        f"Customer ID       : {customer_id}"
    )

    print(
        f"Churn Probability : "
        f"{probability * 100:.2f}%"
    )

    print(
        f"Predicted Churn   : "
        f"{'YES' if prediction else 'NO'}"
    )

    print(
        f"Risk Level        : {risk}"
    )

    print(
        "Recommendation    : "
        f"{generate_recommendation(customer.iloc[0])}"
    )


# ------------------------------------------------------------
# Example Manual Prediction
# ------------------------------------------------------------

print_section("MANUAL CUSTOMER PREDICTION")

example_customer_id = df[
    "customerID"
].iloc[0]

print(
    f"Testing example customer: "
    f"{example_customer_id}"
)

predict_customer(
    example_customer_id
)


# ------------------------------------------------------------
# Final Output
# ------------------------------------------------------------

print_section("CUSTOMER RISK SCORING COMPLETE")

print("Generated files:")

print(
    "- customer_risk_scores.csv"
)

print(
    "- high_risk_customers.csv"
)

print(
    "- risk_summary.csv"
)

print()
print(
    f"Final business threshold: "
    f"{MODEL_THRESHOLD:.2f}"
)

print(
    "Customer risk scoring completed successfully."
)