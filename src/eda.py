"""
Customer Churn Prediction & Business Intelligence System

Exploratory Data Analysis Module
------------------------------------------------------------
Analyzes customer behavior and identifies patterns associated
with customer churn.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "featured_churn_data.csv"
)

REPORTS_DIR = PROJECT_ROOT / "reports"

EDA_DIR = REPORTS_DIR / "eda"


# ============================================================
# LOAD DATA
# ============================================================

def load_data(file_path):
    """Load the feature-engineered dataset."""

    print("\nLoading feature-engineered dataset...")

    df = pd.read_csv(file_path)

    print("Dataset loaded successfully.")
    print(f"Rows: {df.shape[0]}")
    print(f"Columns: {df.shape[1]}")

    return df


# ============================================================
# BASIC DATA ANALYSIS
# ============================================================

def basic_analysis(df):
    """Display basic dataset statistics."""

    print("\n" + "=" * 70)
    print("BASIC DATA ANALYSIS")
    print("=" * 70)

    print("\nDataset shape:")
    print(df.shape)

    print("\nData types:")
    print(df.dtypes)

    print("\nMissing values:")
    print(df.isnull().sum())

    print("\nDuplicate rows:")
    print(df.duplicated().sum())

    print("\nNumerical summary:")
    print(df.describe())


# ============================================================
# CHURN SUMMARY
# ============================================================

def churn_summary(df):
    """Calculate overall churn statistics."""

    print("\n" + "=" * 70)
    print("CHURN SUMMARY")
    print("=" * 70)

    churn_counts = df["Churn"].value_counts()

    total_customers = len(df)

    churned_customers = churn_counts.get(1, 0)

    retained_customers = churn_counts.get(0, 0)

    churn_rate = (
        churned_customers / total_customers
    ) * 100

    print(f"\nTotal customers: {total_customers:,}")
    print(f"Retained customers: {retained_customers:,}")
    print(f"Churned customers: {churned_customers:,}")
    print(f"Overall churn rate: {churn_rate:.2f}%")


# ============================================================
# CREATE EDA DIRECTORY
# ============================================================

def create_output_directory():
    """Create directory for EDA visualizations."""

    EDA_DIR.mkdir(
        parents=True,
        exist_ok=True
    )


# ============================================================
# 1. CHURN DISTRIBUTION
# ============================================================

def plot_churn_distribution(df):
    """Plot overall churn distribution."""

    plt.figure(figsize=(8, 5))

    sns.countplot(
        data=df,
        x="Churn"
    )

    plt.title("Customer Churn Distribution")
    plt.xlabel("Churn")
    plt.ylabel("Number of Customers")

    plt.xticks(
        [0, 1],
        ["No Churn", "Churn"]
    )

    plt.tight_layout()

    plt.savefig(
        EDA_DIR / "01_churn_distribution.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()

    plt.close()


# ============================================================
# 2. CHURN BY CONTRACT
# ============================================================

def plot_churn_by_contract(df):
    """Analyze churn by contract type."""

    churn_rate = (
        df.groupby("Contract")["Churn"]
        .mean()
        .sort_values(ascending=False)
        * 100
    )

    print("\nChurn rate by contract:")
    print(churn_rate)

    plt.figure(figsize=(9, 5))

    sns.barplot(
        x=churn_rate.index,
        y=churn_rate.values
    )

    plt.title("Churn Rate by Contract Type")
    plt.xlabel("Contract Type")
    plt.ylabel("Churn Rate (%)")

    plt.xticks(rotation=15)

    plt.tight_layout()

    plt.savefig(
        EDA_DIR / "02_churn_by_contract.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()

    plt.close()


# ============================================================
# 3. CHURN BY CUSTOMER SEGMENT
# ============================================================

def plot_churn_by_segment(df):
    """Analyze churn by customer segment."""

    segment_order = [
        "New",
        "Established",
        "Long-Term"
    ]

    churn_rate = (
        df.groupby("Customer_Segment")["Churn"]
        .mean()
        .reindex(segment_order)
        * 100
    )

    print("\nChurn rate by customer segment:")
    print(churn_rate)

    plt.figure(figsize=(9, 5))

    sns.barplot(
        x=churn_rate.index,
        y=churn_rate.values
    )

    plt.title("Churn Rate by Customer Segment")
    plt.xlabel("Customer Segment")
    plt.ylabel("Churn Rate (%)")

    plt.tight_layout()

    plt.savefig(
        EDA_DIR / "03_churn_by_segment.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()

    plt.close()


# ============================================================
# 4. TENURE VS CHURN
# ============================================================

def plot_tenure_vs_churn(df):
    """Analyze the relationship between tenure and churn."""

    plt.figure(figsize=(10, 6))

    sns.boxplot(
        data=df,
        x="Churn",
        y="tenure"
    )

    plt.title("Customer Tenure vs Churn")
    plt.xlabel("Churn")
    plt.ylabel("Tenure (Months)")

    plt.xticks(
        [0, 1],
        ["No Churn", "Churn"]
    )

    plt.tight_layout()

    plt.savefig(
        EDA_DIR / "04_tenure_vs_churn.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()

    plt.close()


# ============================================================
# 5. MONTHLY CHARGES VS CHURN
# ============================================================

def plot_monthly_charges_vs_churn(df):
    """Analyze monthly charges by churn status."""

    plt.figure(figsize=(10, 6))

    sns.boxplot(
        data=df,
        x="Churn",
        y="MonthlyCharges"
    )

    plt.title("Monthly Charges vs Churn")
    plt.xlabel("Churn")
    plt.ylabel("Monthly Charges")

    plt.xticks(
        [0, 1],
        ["No Churn", "Churn"]
    )

    plt.tight_layout()

    plt.savefig(
        EDA_DIR / "05_monthly_charges_vs_churn.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()

    plt.close()


# ============================================================
# 6. TOTAL CHARGES VS CHURN
# ============================================================

def plot_total_charges_vs_churn(df):
    """Analyze total charges by churn status."""

    plt.figure(figsize=(10, 6))

    sns.boxplot(
        data=df,
        x="Churn",
        y="TotalCharges"
    )

    plt.title("Total Charges vs Churn")
    plt.xlabel("Churn")
    plt.ylabel("Total Charges")

    plt.xticks(
        [0, 1],
        ["No Churn", "Churn"]
    )

    plt.tight_layout()

    plt.savefig(
        EDA_DIR / "06_total_charges_vs_churn.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()

    plt.close()


# ============================================================
# 7. PAYMENT METHOD
# ============================================================

def plot_churn_by_payment_method(df):
    """Analyze churn by payment method."""

    churn_rate = (
        df.groupby("PaymentMethod")["Churn"]
        .mean()
        .sort_values(ascending=False)
        * 100
    )

    print("\nChurn rate by payment method:")
    print(churn_rate)

    plt.figure(figsize=(11, 6))

    sns.barplot(
        x=churn_rate.values,
        y=churn_rate.index
    )

    plt.title("Churn Rate by Payment Method")
    plt.xlabel("Churn Rate (%)")
    plt.ylabel("Payment Method")

    plt.tight_layout()

    plt.savefig(
        EDA_DIR / "07_churn_by_payment_method.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()

    plt.close()


# ============================================================
# 8. INTERNET SERVICE
# ============================================================

def plot_churn_by_internet_service(df):
    """Analyze churn by internet service type."""

    churn_rate = (
        df.groupby("InternetService")["Churn"]
        .mean()
        .sort_values(ascending=False)
        * 100
    )

    print("\nChurn rate by internet service:")
    print(churn_rate)

    plt.figure(figsize=(9, 5))

    sns.barplot(
        x=churn_rate.index,
        y=churn_rate.values
    )

    plt.title("Churn Rate by Internet Service")
    plt.xlabel("Internet Service")
    plt.ylabel("Churn Rate (%)")

    plt.tight_layout()

    plt.savefig(
        EDA_DIR / "08_churn_by_internet_service.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()

    plt.close()


# ============================================================
# 9. SENIOR CITIZEN
# ============================================================

def plot_churn_by_senior_citizen(df):
    """Analyze churn by senior citizen status."""

    churn_rate = (
        df.groupby("SeniorCitizen")["Churn"]
        .mean()
        * 100
    )

    churn_rate.index = [
        "Non-Senior",
        "Senior"
    ]

    print("\nChurn rate by senior citizen status:")
    print(churn_rate)

    plt.figure(figsize=(8, 5))

    sns.barplot(
        x=churn_rate.index,
        y=churn_rate.values
    )

    plt.title("Churn Rate by Senior Citizen Status")
    plt.xlabel("Customer Group")
    plt.ylabel("Churn Rate (%)")

    plt.tight_layout()

    plt.savefig(
        EDA_DIR / "09_churn_by_senior_citizen.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()

    plt.close()


# ============================================================
# 10. TECH SUPPORT
# ============================================================

def plot_churn_by_tech_support(df):
    """Analyze churn based on technical support."""

    churn_rate = (
        df.groupby("TechSupport")["Churn"]
        .mean()
        .sort_values(ascending=False)
        * 100
    )

    print("\nChurn rate by technical support:")
    print(churn_rate)

    plt.figure(figsize=(9, 5))

    sns.barplot(
        x=churn_rate.index,
        y=churn_rate.values
    )

    plt.title("Churn Rate by Technical Support")
    plt.xlabel("Technical Support")
    plt.ylabel("Churn Rate (%)")

    plt.xticks(rotation=15)

    plt.tight_layout()

    plt.savefig(
        EDA_DIR / "10_churn_by_tech_support.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()

    plt.close()


# ============================================================
# 11. ONLINE SECURITY
# ============================================================

def plot_churn_by_online_security(df):
    """Analyze churn based on online security service."""

    churn_rate = (
        df.groupby("OnlineSecurity")["Churn"]
        .mean()
        .sort_values(ascending=False)
        * 100
    )

    print("\nChurn rate by online security:")
    print(churn_rate)

    plt.figure(figsize=(9, 5))

    sns.barplot(
        x=churn_rate.index,
        y=churn_rate.values
    )

    plt.title("Churn Rate by Online Security")
    plt.xlabel("Online Security")
    plt.ylabel("Churn Rate (%)")

    plt.xticks(rotation=15)

    plt.tight_layout()

    plt.savefig(
        EDA_DIR / "11_churn_by_online_security.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()

    plt.close()


# ============================================================
# 12. SERVICE COUNT
# ============================================================

def plot_churn_by_service_count(df):
    """Analyze churn by number of subscribed services."""

    churn_rate = (
        df.groupby("Service_Count")["Churn"]
        .mean()
        * 100
    )

    print("\nChurn rate by service count:")
    print(churn_rate)

    plt.figure(figsize=(10, 6))

    sns.barplot(
        x=churn_rate.index,
        y=churn_rate.values
    )

    plt.title("Churn Rate by Number of Services")
    plt.xlabel("Number of Services")
    plt.ylabel("Churn Rate (%)")

    plt.tight_layout()

    plt.savefig(
        EDA_DIR / "12_churn_by_service_count.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()

    plt.close()


# ============================================================
# 13. CORRELATION MATRIX
# ============================================================

def plot_correlation_matrix(df):
    """Plot correlation matrix for numerical variables."""

    numeric_columns = df.select_dtypes(
        include=np.number
    ).columns

    correlation = df[numeric_columns].corr()

    plt.figure(figsize=(12, 9))

    sns.heatmap(
        correlation,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        linewidths=0.5
    )

    plt.title("Numerical Feature Correlation Matrix")

    plt.tight_layout()

    plt.savefig(
        EDA_DIR / "13_correlation_matrix.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()

    plt.close()


# ============================================================
# 14. CHURN RATE BY MONTHLY CHARGES
# ============================================================

def create_charge_groups(df):
    """Create monthly charge groups for analysis."""

    df = df.copy()

    df["MonthlyCharge_Group"] = pd.cut(
        df["MonthlyCharges"],
        bins=[
            0,
            40,
            60,
            80,
            100,
            np.inf
        ],
        labels=[
            "≤ $40",
            "$40–60",
            "$60–80",
            "$80–100",
            "> $100"
        ],
        include_lowest=True
    )

    return df


def plot_churn_by_charge_group(df):
    """Analyze churn across monthly charge groups."""

    df = create_charge_groups(df)

    churn_rate = (
        df.groupby(
            "MonthlyCharge_Group",
            observed=True
        )["Churn"]
        .mean()
        * 100
    )

    print("\nChurn rate by monthly charge group:")
    print(churn_rate)

    plt.figure(figsize=(10, 6))

    sns.barplot(
        x=churn_rate.index,
        y=churn_rate.values
    )

    plt.title("Churn Rate by Monthly Charge Group")
    plt.xlabel("Monthly Charge Group")
    plt.ylabel("Churn Rate (%)")

    plt.tight_layout()

    plt.savefig(
        EDA_DIR / "14_churn_by_charge_group.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()

    plt.close()


# ============================================================
# 15. CUSTOMER SEGMENT + CONTRACT
# ============================================================

def plot_segment_contract_churn(df):
    """Analyze churn across customer segments and contracts."""

    analysis = (
        df.groupby(
            ["Customer_Segment", "Contract"]
        )["Churn"]
        .mean()
        .reset_index()
    )

    analysis["Churn"] *= 100

    print("\nChurn rate by customer segment and contract:")
    print(analysis)

    plt.figure(figsize=(11, 6))

    sns.barplot(
        data=analysis,
        x="Customer_Segment",
        y="Churn",
        hue="Contract"
    )

    plt.title(
        "Churn Rate by Customer Segment and Contract"
    )

    plt.xlabel("Customer Segment")
    plt.ylabel("Churn Rate (%)")

    plt.legend(
        title="Contract"
    )

    plt.tight_layout()

    plt.savefig(
        EDA_DIR / "15_segment_contract_churn.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()

    plt.close()


# ============================================================
# MAIN EDA PIPELINE
# ============================================================

def main():

    print("=" * 70)
    print("CUSTOMER CHURN EXPLORATORY DATA ANALYSIS")
    print("=" * 70)

    # Create output directory
    create_output_directory()

    # Load dataset
    df = load_data(DATA_PATH)

    # Basic analysis
    basic_analysis(df)

    # Churn summary
    churn_summary(df)

    print("\nGenerating EDA visualizations...")

    # Generate visualizations
    plot_churn_distribution(df)

    plot_churn_by_contract(df)

    plot_churn_by_segment(df)

    plot_tenure_vs_churn(df)

    plot_monthly_charges_vs_churn(df)

    plot_total_charges_vs_churn(df)

    plot_churn_by_payment_method(df)

    plot_churn_by_internet_service(df)

    plot_churn_by_senior_citizen(df)

    plot_churn_by_tech_support(df)

    plot_churn_by_online_security(df)

    plot_churn_by_service_count(df)

    plot_correlation_matrix(df)

    plot_churn_by_charge_group(df)

    plot_segment_contract_churn(df)

    print("\n" + "=" * 70)
    print("EDA COMPLETE")
    print("=" * 70)

    print(
        f"\nEDA charts saved to:\n{EDA_DIR}"
    )


# ============================================================
# RUN PROGRAM
# ============================================================

if __name__ == "__main__":
    main()