import os
import joblib
import numpy as np
import pandas as pd


# ============================================================
# CUSTOMER CHURN MODEL CONSISTENCY TEST
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

RAW_FILE = os.path.join(
    BASE_DIR,
    "data",
    "raw",
    "WA_Fn-UseC_-Telco-Customer-Churn.csv"
)

FEATURED_FILE = os.path.join(
    BASE_DIR,
    "data",
    "processed",
    "featured_churn_data.csv"
)

MODEL_FILE = os.path.join(
    BASE_DIR,
    "models",
    "best_churn_model.pkl"
)


print("=" * 70)
print("CUSTOMER CHURN MODEL CONSISTENCY TEST")
print("=" * 70)


# ============================================================
# 1. LOAD MODEL
# ============================================================

print("\n[1] Loading trained model...")

model = joblib.load(MODEL_FILE)

print("Model loaded successfully.")
print("Model type:", type(model).__name__)


# ============================================================
# 2. LOAD RAW DATA
# ============================================================

print("\n[2] Loading raw customer CSV...")

raw_df = pd.read_csv(RAW_FILE)

print("Raw dataset shape:", raw_df.shape)


# ============================================================
# 3. RAW DATA CLEANING
# ============================================================

print("\n[3] Cleaning raw data...")

raw_df = raw_df.copy()

raw_df["TotalCharges"] = pd.to_numeric(
    raw_df["TotalCharges"],
    errors="coerce"
)

raw_df["TotalCharges"] = raw_df["TotalCharges"].fillna(
    raw_df["MonthlyCharges"] * raw_df["tenure"]
)

if "Churn" in raw_df.columns:
    raw_df["Churn"] = raw_df["Churn"].map(
        {"Yes": 1, "No": 0}
    )

print("Raw data cleaning completed.")


# ============================================================
# 4. FEATURE ENGINEERING
# ============================================================

print("\n[4] Applying feature engineering...")

raw_df["Customer_Lifetime_Months"] = raw_df["tenure"]

raw_df["Average_Monthly_Spend"] = (
    raw_df["TotalCharges"] / raw_df["tenure"]
)

raw_df["Average_Monthly_Spend"] = (
    raw_df["Average_Monthly_Spend"]
    .replace([float("inf"), -float("inf")], np.nan)
    .fillna(raw_df["MonthlyCharges"])
)


# Service count

raw_df["Service_Count"] = 0

raw_df["Service_Count"] += (
    raw_df["PhoneService"] == "Yes"
).astype(int)

raw_df["Service_Count"] += (
    raw_df["MultipleLines"] == "Yes"
).astype(int)

for column in [
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies"
]:
    raw_df["Service_Count"] += (
        raw_df[column] == "Yes"
    ).astype(int)


# Paperless billing

raw_df["Paperless_Billing_Flag"] = (
    raw_df["PaperlessBilling"] == "Yes"
).astype(int)


# Automatic payment

automatic_payment_methods = [
    "Bank transfer (automatic)",
    "Credit card (automatic)"
]

raw_df["Automatic_Payment_Flag"] = (
    raw_df["PaymentMethod"]
    .isin(automatic_payment_methods)
    .astype(int)
)


# Short tenure

raw_df["Short_Tenure_Flag"] = (
    raw_df["tenure"] <= 12
).astype(int)


# High monthly charge

monthly_charge_median = raw_df["MonthlyCharges"].median()

raw_df["High_Monthly_Charge_Flag"] = (
    raw_df["MonthlyCharges"] > monthly_charge_median
).astype(int)


# Customer segment

def assign_customer_segment(tenure):

    if tenure <= 12:
        return "New"

    elif tenure <= 36:
        return "Established"

    else:
        return "Long-Term"


raw_df["Customer_Segment"] = (
    raw_df["tenure"].apply(assign_customer_segment)
)


print("Feature engineering completed.")

print(
    "Raw dataset after feature engineering:",
    raw_df.shape
)


# ============================================================
# 5. LOAD FEATURED DATASET
# ============================================================

print("\n[5] Loading existing feature-engineered dataset...")

featured_df = pd.read_csv(FEATURED_FILE)

print(
    "Feature-engineered dataset shape:",
    featured_df.shape
)


# ============================================================
# 6. PREPARE MODEL INPUT
# ============================================================

print("\n[6] Preparing model inputs...")

NON_FEATURE_COLUMNS = [
    "customerID",
    "Churn"
]


raw_X = raw_df.drop(
    columns=[
        column
        for column in NON_FEATURE_COLUMNS
        if column in raw_df.columns
    ]
)

featured_X = featured_df.drop(
    columns=[
        column
        for column in NON_FEATURE_COLUMNS
        if column in featured_df.columns
    ]
)


print("Raw model input shape:", raw_X.shape)

print(
    "Featured model input shape:",
    featured_X.shape
)


# ============================================================
# 7. CHECK FEATURE COLUMNS
# ============================================================

print("\n[7] Comparing feature columns...")

raw_columns = set(raw_X.columns)

featured_columns = set(featured_X.columns)

only_in_raw = sorted(
    raw_columns - featured_columns
)

only_in_featured = sorted(
    featured_columns - raw_columns
)


if not only_in_raw and not only_in_featured:

    print("✓ Feature columns match.")

else:

    print("WARNING: Feature columns do not match.")

    if only_in_raw:
        print("\nOnly in raw:")
        print(only_in_raw)

    if only_in_featured:
        print("\nOnly in featured:")
        print(only_in_featured)


# ============================================================
# 8. GENERATE PREDICTIONS
# ============================================================

print("\n[8] Generating predictions...")

raw_probabilities = model.predict_proba(
    raw_X
)[:, 1]

featured_probabilities = model.predict_proba(
    featured_X
)[:, 1]


# ============================================================
# 9. COMPARE PROBABILITIES
# ============================================================

print("\n[9] Comparing churn probabilities...")

difference = np.abs(
    raw_probabilities -
    featured_probabilities
)

print(
    "Maximum probability difference:",
    difference.max()
)

print(
    "Mean probability difference:",
    difference.mean()
)

print(
    "Number of different probabilities:",
    np.sum(difference > 1e-10)
)


# ============================================================
# 10. COMPARE PREDICTIONS
# ============================================================

raw_predictions = (
    raw_probabilities >= 0.30
).astype(int)

featured_predictions = (
    featured_probabilities >= 0.30
).astype(int)


prediction_difference = (
    raw_predictions != featured_predictions
)


print("\n[10] Comparing churn classifications...")

print(
    "Raw predicted churn:",
    raw_predictions.sum()
)

print(
    "Featured predicted churn:",
    featured_predictions.sum()
)

print(
    "Different classifications:",
    prediction_difference.sum()
)


# ============================================================
# 11. SUMMARY STATISTICS
# ============================================================

print("\n[11] Probability summary")

print("-" * 50)

print(
    f"Raw average probability:      "
    f"{raw_probabilities.mean():.6f}"
)

print(
    f"Featured average probability: "
    f"{featured_probabilities.mean():.6f}"
)

print(
    f"Raw HIGH risk (>=70%):        "
    f"{(raw_probabilities >= 0.70).sum()}"
)

print(
    f"Featured HIGH risk (>=70%):   "
    f"{(featured_probabilities >= 0.70).sum()}"
)


# ============================================================
# 12. FINAL RESULT
# ============================================================

print("\n" + "=" * 70)
print("FINAL RESULT")
print("=" * 70)

if (
    np.allclose(
        raw_probabilities,
        featured_probabilities,
        atol=1e-10
    )
    and
    prediction_difference.sum() == 0
):

    print(
        "\n✓ PASS: Raw and feature-engineered datasets"
        " produce identical predictions."
    )

    print(
        "\nThe trained model and feature-engineering"
        " pipeline are consistent."
    )

else:

    print(
        "\n✗ WARNING: Predictions are different."
    )

    print(
        "\nThe raw and feature-engineered processing"
        " pipelines are not producing identical inputs."
    )


print("\n" + "=" * 70)
print("TEST COMPLETE")
print("=" * 70)