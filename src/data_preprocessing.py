"""
Customer Churn Prediction & Business Intelligence System

Data Preprocessing Module
------------------------------------------------------------
Loads the raw Telco Customer Churn dataset, cleans the data,
performs basic preprocessing, and saves the processed dataset.
"""

import pandas as pd
from pathlib import Path


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

RAW_DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
)

PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"

PROCESSED_DATA_PATH = (
    PROCESSED_DATA_DIR
    / "cleaned_churn_data.csv"
)


# ============================================================
# LOAD DATA
# ============================================================

def load_data(file_path):
    """Load the raw CSV dataset."""

    print("\nLoading dataset...")

    df = pd.read_csv(file_path)

    print(f"Dataset loaded successfully.")
    print(f"Rows: {df.shape[0]}")
    print(f"Columns: {df.shape[1]}")

    return df


# ============================================================
# DATA CLEANING
# ============================================================

def clean_data(df):
    """Clean and prepare the customer churn dataset."""

    print("\nStarting data cleaning...")

    df = df.copy()

    # --------------------------------------------------------
    # Remove duplicate rows
    # --------------------------------------------------------

    duplicate_count = df.duplicated().sum()

    print(f"Duplicate rows found: {duplicate_count}")

    if duplicate_count > 0:
        df = df.drop_duplicates()

    # --------------------------------------------------------
    # Clean TotalCharges
    # --------------------------------------------------------
    # TotalCharges may contain blank strings.
    # Convert them to numeric values.
    # Invalid values become NaN.

    df["TotalCharges"] = pd.to_numeric(
        df["TotalCharges"],
        errors="coerce"
    )

    total_charges_missing = df["TotalCharges"].isna().sum()

    print(
        f"Missing TotalCharges after conversion: "
        f"{total_charges_missing}"
    )

    # --------------------------------------------------------
    # Handle missing TotalCharges
    # --------------------------------------------------------
    # For customers with zero tenure, TotalCharges can be
    # missing. Using MonthlyCharges * tenure gives a sensible
    # estimate for those records.

    if total_charges_missing > 0:

        df["TotalCharges"] = df["TotalCharges"].fillna(
            df["MonthlyCharges"] * df["tenure"]
        )

    # --------------------------------------------------------
    # Convert target variable
    # --------------------------------------------------------

    df["Churn"] = df["Churn"].map({
        "Yes": 1,
        "No": 0
    })

    # --------------------------------------------------------
    # Final missing-value check
    # --------------------------------------------------------

    remaining_missing = df.isnull().sum().sum()

    print(
        f"Remaining missing values: "
        f"{remaining_missing}"
    )

    # --------------------------------------------------------
    # Reset index
    # --------------------------------------------------------

    df = df.reset_index(drop=True)

    return df


# ============================================================
# SAVE PROCESSED DATA
# ============================================================

def save_processed_data(df, output_path):
    """Save the cleaned dataset."""

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    df.to_csv(
        output_path,
        index=False
    )

    print("\nProcessed dataset saved successfully.")
    print(f"Location: {output_path}")


# ============================================================
# MAIN PIPELINE
# ============================================================

def main():

    print("=" * 70)
    print("CUSTOMER CHURN DATA PREPROCESSING")
    print("=" * 70)

    # Load raw data
    df = load_data(RAW_DATA_PATH)

    # Clean data
    df_cleaned = clean_data(df)

    # Save processed data
    save_processed_data(
        df_cleaned,
        PROCESSED_DATA_PATH
    )

    # --------------------------------------------------------
    # Final summary
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("PREPROCESSING COMPLETE")
    print("=" * 70)

    print(f"Final rows: {df_cleaned.shape[0]}")
    print(f"Final columns: {df_cleaned.shape[1]}")

    print("\nChurn distribution:")

    print(
        df_cleaned["Churn"]
        .value_counts()
        .sort_index()
    )

    print("\nData types:")

    print(df_cleaned.dtypes)

    print("\nFirst 5 processed records:")

    print(df_cleaned.head())


# ============================================================
# RUN PROGRAM
# ============================================================

if __name__ == "__main__":
    main()