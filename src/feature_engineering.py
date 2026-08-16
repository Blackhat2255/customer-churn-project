"""
Customer Churn Prediction & Business Intelligence System

Feature Engineering Module
------------------------------------------------------------
Creates business-oriented features from the cleaned Telco
Customer Churn dataset.
"""

import pandas as pd
from pathlib import Path


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

PROCESSED_DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "cleaned_churn_data.csv"
)

FEATURED_DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "featured_churn_data.csv"
)


# ============================================================
# LOAD CLEANED DATA
# ============================================================

def load_cleaned_data(file_path):
    """Load the cleaned customer churn dataset."""

    print("\nLoading cleaned dataset...")

    df = pd.read_csv(file_path)

    print("Cleaned dataset loaded successfully.")
    print(f"Rows: {df.shape[0]}")
    print(f"Columns: {df.shape[1]}")

    return df


# ============================================================
# FEATURE ENGINEERING
# ============================================================

def create_features(df):
    """Create additional features for churn analysis."""

    print("\nStarting feature engineering...")

    df = df.copy()

    # ========================================================
    # 1. CUSTOMER LIFETIME
    # ========================================================

    # The original tenure column represents the number of
    # months the customer has been with the company.

    df["Customer_Lifetime_Months"] = df["tenure"]

    # ========================================================
    # 2. AVERAGE MONTHLY SPEND
    # ========================================================

    # Calculate average monthly spending based on the
    # customer's total charges and tenure.

    df["Average_Monthly_Spend"] = df["TotalCharges"] / df["tenure"]

    # For customers with zero tenure, use their current
    # MonthlyCharges instead of dividing by zero.

    df["Average_Monthly_Spend"] = (
        df["Average_Monthly_Spend"]
        .replace([float("inf"), -float("inf")], pd.NA)
        .fillna(df["MonthlyCharges"])
    )

    # ========================================================
    # 3. SERVICE COUNT
    # ========================================================

    service_columns = [
        "PhoneService",
        "MultipleLines",
        "OnlineSecurity",
        "OnlineBackup",
        "DeviceProtection",
        "TechSupport",
        "StreamingTV",
        "StreamingMovies"
    ]

    df["Service_Count"] = 0

    # PhoneService
    df["Service_Count"] += (
        df["PhoneService"] == "Yes"
    ).astype(int)

    # MultipleLines
    df["Service_Count"] += (
        df["MultipleLines"] == "Yes"
    ).astype(int)

    # Internet-related services
    for column in [
        "OnlineSecurity",
        "OnlineBackup",
        "DeviceProtection",
        "TechSupport",
        "StreamingTV",
        "StreamingMovies"
    ]:
        df["Service_Count"] += (
            df[column] == "Yes"
        ).astype(int)

    # ========================================================
    # 4. PAPERLESS BILLING FLAG
    # ========================================================

    df["Paperless_Billing_Flag"] = (
        df["PaperlessBilling"] == "Yes"
    ).astype(int)

    # ========================================================
    # 5. AUTOMATIC PAYMENT FLAG
    # ========================================================

    automatic_payment_methods = [
        "Bank transfer (automatic)",
        "Credit card (automatic)"
    ]

    df["Automatic_Payment_Flag"] = (
        df["PaymentMethod"]
        .isin(automatic_payment_methods)
        .astype(int)
    )

    # ========================================================
    # 6. SHORT TENURE FLAG
    # ========================================================

    # Customers with 12 months or less are considered
    # relatively new customers.

    df["Short_Tenure_Flag"] = (
        df["tenure"] <= 12
    ).astype(int)

    # ========================================================
    # 7. HIGH MONTHLY CHARGE FLAG
    # ========================================================

    monthly_charge_median = df["MonthlyCharges"].median()

    df["High_Monthly_Charge_Flag"] = (
        df["MonthlyCharges"] > monthly_charge_median
    ).astype(int)

    # ========================================================
    # 8. CUSTOMER SEGMENT
    # ========================================================

    def assign_customer_segment(tenure):
        if tenure <= 12:
            return "New"
        elif tenure <= 36:
            return "Established"
        else:
            return "Long-Term"

    df["Customer_Segment"] = (
        df["tenure"]
        .apply(assign_customer_segment)
    )

    print("\nFeatures created successfully.")

    return df


# ============================================================
# DISPLAY FEATURE SUMMARY
# ============================================================

def display_feature_summary(df):
    """Display information about the engineered features."""

    engineered_features = [
        "Customer_Lifetime_Months",
        "Average_Monthly_Spend",
        "Service_Count",
        "Paperless_Billing_Flag",
        "Automatic_Payment_Flag",
        "Short_Tenure_Flag",
        "High_Monthly_Charge_Flag",
        "Customer_Segment"
    ]

    print("\n" + "=" * 70)
    print("ENGINEERED FEATURES")
    print("=" * 70)

    for feature in engineered_features:
        print(f"- {feature}")

    print("\nFeature preview:")

    print(
        df[engineered_features].head()
    )

    print("\nCustomer segment distribution:")

    print(
        df["Customer_Segment"]
        .value_counts()
    )


# ============================================================
# SAVE FEATURED DATA
# ============================================================

def save_featured_data(df, output_path):
    """Save the feature-engineered dataset."""

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    df.to_csv(
        output_path,
        index=False
    )

    print("\nFeature-engineered dataset saved successfully.")
    print(f"Location: {output_path}")


# ============================================================
# MAIN PIPELINE
# ============================================================

def main():

    print("=" * 70)
    print("CUSTOMER CHURN FEATURE ENGINEERING")
    print("=" * 70)

    # Load cleaned dataset
    df = load_cleaned_data(
        PROCESSED_DATA_PATH
    )

    # Create features
    df_featured = create_features(df)

    # Display feature information
    display_feature_summary(
        df_featured
    )

    # Save featured dataset
    save_featured_data(
        df_featured,
        FEATURED_DATA_PATH
    )

    # Final summary
    print("\n" + "=" * 70)
    print("FEATURE ENGINEERING COMPLETE")
    print("=" * 70)

    print(
        f"Original columns: {df.shape[1]}"
    )

    print(
        f"Final columns: {df_featured.shape[1]}"
    )

    print(
        f"Rows: {df_featured.shape[0]}"
    )


# ============================================================
# RUN PROGRAM
# ============================================================

if __name__ == "__main__":
    main()