# ============================================================
# CUSTOMER CHURN PREDICTION & BUSINESS INTELLIGENCE SYSTEM
# Streamlit Dashboard
# ============================================================

import os
import warnings

import numpy as np
import pandas as pd
import streamlit as st

warnings.filterwarnings("ignore")


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Customer Churn BI System",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# PROJECT PATHS
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "best_churn_model.pkl"
)

RAW_DATA_PATH = os.path.join(
    BASE_DIR,
    "data",
    "raw",
    "WA_Fn-UseC_-Telco-Customer-Churn.csv"
)

CLEANED_DATA_PATH = os.path.join(
    BASE_DIR,
    "data",
    "processed",
    "cleaned_churn_data.csv"
)

FEATURED_DATA_PATH = os.path.join(
    BASE_DIR,
    "data",
    "processed",
    "featured_churn_data.csv"
)

MODEL_COMPARISON_PATH = os.path.join(
    BASE_DIR,
    "reports",
    "model_comparison_results.csv"
)

FEATURE_IMPORTANCE_PATH = os.path.join(
    BASE_DIR,
    "reports",
    "explainability",
    "aggregated_feature_importance.csv"
)

PERMUTATION_IMPORTANCE_PATH = os.path.join(
    BASE_DIR,
    "reports",
    "explainability",
    "permutation_feature_importance.csv"
)

EVALUATION_DIR = os.path.join(
    BASE_DIR,
    "reports",
    "evaluation"
)

EVALUATION_SUMMARY_PATH = os.path.join(
    EVALUATION_DIR,
    "evaluation_summary.csv"
)

CONFUSION_MATRIX_PATH = os.path.join(
    EVALUATION_DIR,
    "confusion_matrix.png"
)

ROC_CURVE_PATH = os.path.join(
    EVALUATION_DIR,
    "roc_curve.png"
)

PR_CURVE_PATH = os.path.join(
    EVALUATION_DIR,
    "precision_recall_curve.png"
)

THRESHOLD_ANALYSIS_PATH = os.path.join(
    EVALUATION_DIR,
    "threshold_analysis.png"
)


# ============================================================
# PROJECT SETTINGS
# ============================================================

MODEL_NAME = "Gradient Boosting Classifier"

CHURN_THRESHOLD = 0.30

HIGH_RISK_THRESHOLD = 0.70


# ============================================================
# REQUIRED RAW COLUMNS
# ============================================================

REQUIRED_RAW_COLUMNS = [
    "customerID",
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "tenure",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
    "MonthlyCharges",
    "TotalCharges"
]


# ============================================================
# ENGINEERED COLUMNS
# ============================================================

ENGINEERED_COLUMNS = [
    "Customer_Lifetime_Months",
    "Average_Monthly_Spend",
    "Service_Count",
    "Paperless_Billing_Flag",
    "Automatic_Payment_Flag",
    "Short_Tenure_Flag",
    "High_Monthly_Charge_Flag",
    "Customer_Segment"
]


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def format_currency(value):

    try:
        return f"${value:,.0f}"

    except Exception:
        return "$0"


def assign_risk_level(probability):

    if probability >= HIGH_RISK_THRESHOLD:
        return "HIGH"

    elif probability >= CHURN_THRESHOLD:
        return "MEDIUM"

    return "LOW"


def normalize_churn_label(value):

    if pd.isna(value):
        return np.nan

    if isinstance(value, str):

        value = value.strip().lower()

        if value in [
            "yes",
            "y",
            "1",
            "true",
            "churn"
        ]:
            return 1

        if value in [
            "no",
            "n",
            "0",
            "false",
            "not churn"
        ]:
            return 0

    try:
        return int(value)

    except Exception:
        return np.nan


# ============================================================
# LOAD TRAINED MODEL
# ============================================================

@st.cache_resource
def load_model():

    if not os.path.exists(MODEL_PATH):
        return None

    try:

        import joblib

        return joblib.load(
            MODEL_PATH
        )

    except Exception as error:

        st.error(
            f"Model loading failed: {error}"
        )

        return None


# ============================================================
# LOAD CSV
# ============================================================

def load_csv(file):

    try:

        return pd.read_csv(file)

    except Exception as error:

        st.error(
            f"Unable to read CSV file: {error}"
        )

        return None


# ============================================================
# DATA CLEANING
# ============================================================

def clean_uploaded_data(df):

    df = df.copy()

    df.columns = (
        df.columns
        .astype(str)
        .str.strip()
    )

    if "tenure" in df.columns:

        df["tenure"] = pd.to_numeric(
            df["tenure"],
            errors="coerce"
        )

    if "MonthlyCharges" in df.columns:

        df["MonthlyCharges"] = pd.to_numeric(
            df["MonthlyCharges"],
            errors="coerce"
        )

    if "TotalCharges" in df.columns:

        df["TotalCharges"] = pd.to_numeric(
            df["TotalCharges"],
            errors="coerce"
        )

    if (
        "TotalCharges" in df.columns
        and "MonthlyCharges" in df.columns
        and "tenure" in df.columns
    ):

        df["TotalCharges"] = (
            df["TotalCharges"]
            .fillna(
                df["MonthlyCharges"]
                * df["tenure"]
            )
        )

    if "Churn" in df.columns:

        df["Churn"] = (
            df["Churn"]
            .apply(
                normalize_churn_label
            )
        )

    df = df.drop_duplicates()

    return df


# ============================================================
# FEATURE ENGINEERING
# ============================================================

def create_features(df):

    df = df.copy()

    # --------------------------------------------------------
    # Customer Lifetime
    # --------------------------------------------------------

    if "tenure" in df.columns:

        df[
            "Customer_Lifetime_Months"
        ] = df["tenure"]

    # --------------------------------------------------------
    # Average Monthly Spend
    # --------------------------------------------------------

    if all(
        column in df.columns
        for column in [
            "TotalCharges",
            "tenure",
            "MonthlyCharges"
        ]
    ):

        safe_tenure = (
            df["tenure"]
            .replace(0, np.nan)
        )

        df[
            "Average_Monthly_Spend"
        ] = (
            df["TotalCharges"]
            / safe_tenure
        )

        df[
            "Average_Monthly_Spend"
        ] = (
            df[
                "Average_Monthly_Spend"
            ]
            .replace(
                [np.inf, -np.inf],
                np.nan
            )
            .fillna(
                df["MonthlyCharges"]
            )
        )

    # --------------------------------------------------------
    # Service Count
    # --------------------------------------------------------

    df["Service_Count"] = 0

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

    for column in service_columns:

        if column in df.columns:

            df["Service_Count"] += (
                df[column] == "Yes"
            ).astype(int)

    # --------------------------------------------------------
    # Paperless Billing
    # --------------------------------------------------------

    if "PaperlessBilling" in df.columns:

        df[
            "Paperless_Billing_Flag"
        ] = (
            df["PaperlessBilling"] == "Yes"
        ).astype(int)

    # --------------------------------------------------------
    # Automatic Payment
    # --------------------------------------------------------

    if "PaymentMethod" in df.columns:

        automatic_methods = [
            "Bank transfer (automatic)",
            "Credit card (automatic)"
        ]

        df[
            "Automatic_Payment_Flag"
        ] = (
            df["PaymentMethod"]
            .isin(automatic_methods)
            .astype(int)
        )

    # --------------------------------------------------------
    # Short Tenure
    # --------------------------------------------------------

    if "tenure" in df.columns:

        df[
            "Short_Tenure_Flag"
        ] = (
            df["tenure"] <= 12
        ).astype(int)

    # --------------------------------------------------------
    # High Monthly Charge
    # --------------------------------------------------------

    if "MonthlyCharges" in df.columns:

        median_charge = (
            df["MonthlyCharges"]
            .median()
        )

        df[
            "High_Monthly_Charge_Flag"
        ] = (
            df["MonthlyCharges"]
            > median_charge
        ).astype(int)

    # --------------------------------------------------------
    # Customer Segment
    # --------------------------------------------------------

    if "tenure" in df.columns:

        def segment(tenure):

            if tenure <= 12:
                return "New"

            elif tenure <= 36:
                return "Established"

            return "Long-Term"

        df[
            "Customer_Segment"
        ] = (
            df["tenure"]
            .apply(segment)
        )

    return df


# ============================================================
# RETENTION RECOMMENDATION
# ============================================================

def generate_recommendation(row):

    recommendations = []

    if (
        "Contract" in row.index
        and row["Contract"]
        == "Month-to-month"
    ):

        recommendations.append(
            "Offer a long-term contract incentive"
        )

    if (
        "tenure" in row.index
        and row["tenure"] <= 12
    ):

        recommendations.append(
            "Provide new-customer retention support"
        )

    if (
        "MonthlyCharges" in row.index
        and row["MonthlyCharges"] >= 80
    ):

        recommendations.append(
            "Review pricing and offer a suitable discount"
        )

    if (
        "TechSupport" in row.index
        and row["TechSupport"] == "No"
    ):

        recommendations.append(
            "Offer technical support service"
        )

    if (
        "OnlineSecurity" in row.index
        and row["OnlineSecurity"] == "No"
    ):

        recommendations.append(
            "Offer online security package"
        )

    if (
        "InternetService" in row.index
        and row["InternetService"]
        == "Fiber optic"
    ):

        recommendations.append(
            "Review fiber-service satisfaction"
        )

    if (
        "PaymentMethod" in row.index
        and row["PaymentMethod"]
        == "Electronic check"
    ):

        recommendations.append(
            "Encourage automatic payment"
        )

    if not recommendations:

        return (
            "Monitor customer and maintain "
            "current engagement."
        )

    return "; ".join(
        recommendations
    )


# ============================================================
# INDIVIDUAL RISK FACTORS
# ============================================================

def identify_risk_factors(row):

    factors = []

    if (
        "Contract" in row.index
        and row["Contract"]
        == "Month-to-month"
    ):

        factors.append(
            "Month-to-month contract"
        )

    if (
        "tenure" in row.index
        and row["tenure"] <= 12
    ):

        factors.append(
            "Short customer tenure"
        )

    if (
        "MonthlyCharges" in row.index
        and row["MonthlyCharges"] >= 80
    ):

        factors.append(
            "High monthly charges"
        )

    if (
        "TechSupport" in row.index
        and row["TechSupport"] == "No"
    ):

        factors.append(
            "No technical support"
        )

    if (
        "OnlineSecurity" in row.index
        and row["OnlineSecurity"] == "No"
    ):

        factors.append(
            "No online security service"
        )

    if (
        "InternetService" in row.index
        and row["InternetService"]
        == "Fiber optic"
    ):

        factors.append(
            "Fiber optic internet service"
        )

    if (
        "PaymentMethod" in row.index
        and row["PaymentMethod"]
        == "Electronic check"
    ):

        factors.append(
            "Electronic check payment method"
        )

    if not factors:

        factors.append(
            "No major predefined risk factor identified"
        )

    return factors


# ============================================================
# SCORE DATA
# ============================================================

def score_dataframe(
    df,
    model
):

    result = df.copy()

    excluded_columns = {
        "customerID",
        "Churn",
        "Churn_Probability",
        "Predicted_Churn",
        "Risk_Level",
        "Retention_Recommendation"
    }

    model_input = result.drop(
        columns=[
            column
            for column
            in excluded_columns
            if column in result.columns
        ],
        errors="ignore"
    )

    probabilities = (
        model.predict_proba(
            model_input
        )[:, 1]
    )

    result[
        "Churn_Probability"
    ] = probabilities

    result[
        "Predicted_Churn"
    ] = (
        probabilities
        >= CHURN_THRESHOLD
    ).astype(int)

    result[
        "Risk_Level"
    ] = [
        assign_risk_level(
            probability
        )
        for probability
        in probabilities
    ]

    result[
        "Retention_Recommendation"
    ] = (
        result.apply(
            generate_recommendation,
            axis=1
        )
    )

    return result


# ============================================================
# REVENUE RISK
# ============================================================

def calculate_revenue_risk(df):

    if "MonthlyCharges" not in df.columns:

        return {
            "monthly": 0,
            "annual": 0,
            "weighted": 0
        }

    high_risk = df[
        df["Risk_Level"] == "HIGH"
    ]

    monthly = (
        high_risk[
            "MonthlyCharges"
        ].sum()
    )

    annual = monthly * 12

    weighted = (
        (
            df["MonthlyCharges"]
            * df["Churn_Probability"]
        ).sum()
        * 12
    )

    return {
        "monthly": monthly,
        "annual": annual,
        "weighted": weighted
    }


# ============================================================
# LOAD MAIN DATA
# ============================================================

@st.cache_data
def load_main_data():

    # First choice: feature-engineered dataset

    if os.path.exists(
        FEATURED_DATA_PATH
    ):

        try:

            return pd.read_csv(
                FEATURED_DATA_PATH
            )

        except Exception:
            pass

    # Second choice: cleaned dataset

    if os.path.exists(
        CLEANED_DATA_PATH
    ):

        try:

            df = pd.read_csv(
                CLEANED_DATA_PATH
            )

            return create_features(
                df
            )

        except Exception:
            pass

    # Third choice: raw dataset

    if os.path.exists(
        RAW_DATA_PATH
    ):

        try:

            df = pd.read_csv(
                RAW_DATA_PATH
            )

            df = clean_uploaded_data(
                df
            )

            return create_features(
                df
            )

        except Exception:
            pass

    return None


# ============================================================
# DETECT CSV TYPE
# ============================================================

def detect_csv_type(df):

    columns = set(
        df.columns
    )

    # Feature-engineered

    if set(
        ENGINEERED_COLUMNS
    ).issubset(columns):

        return "featured"

    # Cleaned

    if (
        set(
            REQUIRED_RAW_COLUMNS
        ).issubset(columns)
        and "Churn" in columns
    ):

        churn_values = (
            df["Churn"]
            .dropna()
            .astype(str)
            .str.strip()
            .str.lower()
            .unique()
        )

        if set(
            churn_values
        ).issubset(
            {
                "0",
                "1",
                "0.0",
                "1.0"
            }
        ):

            return "cleaned"

    # Raw

    if set(
        REQUIRED_RAW_COLUMNS
    ).issubset(columns):

        return "raw"

    return "unsupported"


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title(
    "📊 Customer Churn BI"
)

st.sidebar.write(
    "Customer Churn Prediction & "
    "Business Intelligence System"
)

st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigation",
    [
        "Executive Overview",
        "Customer Analysis",
        "Churn Analysis",
        "Risk & Retention",
        "CSV Prediction",
        "Individual Customer Prediction",
        "Model Insights"
    ]
)

st.sidebar.markdown("---")

st.sidebar.write(
    f"**Model:** {MODEL_NAME}"
)

st.sidebar.write(
    f"**Churn Threshold:** "
    f"{CHURN_THRESHOLD:.0%}"
)

st.sidebar.write(
    f"**High-Risk Threshold:** "
    f"{HIGH_RISK_THRESHOLD:.0%}"
)

st.sidebar.info(
    "The dashboard uses the existing trained "
    "model and does not retrain it."
)


# ============================================================
# LOAD MODEL AND DATA
# ============================================================

model = load_model()

if model is None:

    st.error(
        "The trained model could not be loaded."
    )

    st.stop()


main_data = load_main_data()

if main_data is None:

    st.error(
        "Project dataset could not be loaded."
    )

    st.stop()


# ============================================================
# MAIN HEADER
# ============================================================

st.title(
    "📊 Customer Churn Prediction "
    "& Business Intelligence System"
)

st.caption(
    "Machine Learning • Business Analytics • "
    "Customer Retention Intelligence"
)


# ============================================================
# EXECUTIVE OVERVIEW
# ============================================================

if page == "Executive Overview":

    st.header(
        "Executive Overview"
    )

    scored = score_dataframe(
        main_data,
        model
    )

    total_customers = len(
        scored
    )

    # Actual historical churn

    if "Churn" in scored.columns:

        actual_churn_rate = (
            pd.to_numeric(
                scored["Churn"],
                errors="coerce"
            )
            .mean()
            * 100
        )

    else:

        actual_churn_rate = 0

    predicted_churn_count = (
        scored[
            "Predicted_Churn"
        ].sum()
    )

    predicted_churn_rate = (
        scored[
            "Predicted_Churn"
        ].mean()
        * 100
    )

    high_risk_count = (
        scored[
            "Risk_Level"
        ]
        .eq("HIGH")
        .sum()
    )

    average_probability = (
        scored[
            "Churn_Probability"
        ].mean()
        * 100
    )

    revenue = calculate_revenue_risk(
        scored
    )

    # --------------------------------------------------------
    # KPI CARDS
    # --------------------------------------------------------

    col1, col2, col3, col4, col5 = (
        st.columns(5)
    )

    col1.metric(
        "Total Customers",
        f"{total_customers:,}"
    )

    col2.metric(
        "Actual Churn Rate",
        f"{actual_churn_rate:.1f}%"
    )

    col3.metric(
        "Predicted Churn",
        f"{predicted_churn_count:,}"
    )

    col4.metric(
        "HIGH Risk Customers",
        f"{high_risk_count:,}"
    )

    col5.metric(
        "Avg. Churn Probability",
        f"{average_probability:.1f}%"
    )

    st.markdown("---")

    # --------------------------------------------------------
    # PREDICTED CHURN RATE
    # --------------------------------------------------------

    col1, col2 = (
        st.columns(2)
    )

    col1.metric(
        "Predicted Churn Rate",
        f"{predicted_churn_rate:.1f}%"
    )

    col2.metric(
        "Total Customers",
        f"{total_customers:,}"
    )

    # --------------------------------------------------------
    # REVENUE RISK
    # --------------------------------------------------------

    st.subheader(
        "Estimated Financial Impact"
    )

    col1, col2, col3 = (
        st.columns(3)
    )

    col1.metric(
        "HIGH-Risk Monthly Revenue",
        format_currency(
            revenue["monthly"]
        )
    )

    col2.metric(
        "HIGH-Risk Annual Revenue",
        format_currency(
            revenue["annual"]
        )
    )

    col3.metric(
        "Probability-Weighted Annual Risk",
        format_currency(
            revenue["weighted"]
        )
    )

    st.markdown("---")

    # --------------------------------------------------------
    # RISK DISTRIBUTION
    # --------------------------------------------------------

    st.subheader(
        "Customer Risk Distribution"
    )

    risk_distribution = (
        scored[
            "Risk_Level"
        ]
        .value_counts()
        .reindex(
            [
                "LOW",
                "MEDIUM",
                "HIGH"
            ],
            fill_value=0
        )
    )

    st.bar_chart(
        risk_distribution
    )

    # --------------------------------------------------------
    # ACTUAL VS PREDICTED
    # --------------------------------------------------------

    if "Churn" in scored.columns:

        st.subheader(
            "Actual vs Predicted Churn"
        )

        actual_count = (
            scored["Churn"]
            .sum()
        )

        predicted_count = (
            scored[
                "Predicted_Churn"
            ].sum()
        )

        comparison = pd.Series(
            {
                "Actual Churn":
                    actual_count,
                "Predicted Churn":
                    predicted_count
            }
        )

        st.bar_chart(
            comparison
        )

    # --------------------------------------------------------
    # MODEL INFORMATION
    # --------------------------------------------------------

    st.subheader(
        "Model Information"
    )

    st.write(
        f"**Selected Model:** {MODEL_NAME}"
    )

    st.write(
        f"**Churn Classification Threshold:** "
        f"{CHURN_THRESHOLD:.0%}"
    )

    st.write(
        f"**High-Risk Threshold:** "
        f"{HIGH_RISK_THRESHOLD:.0%}"
    )

    st.info(
        "The model is loaded from the existing "
        "trained model file. Prediction does not "
        "retrain the model."
    )


# ============================================================
# CUSTOMER ANALYSIS
# ============================================================

elif page == "Customer Analysis":

    st.header(
        "Customer Analysis"
    )

    scored = score_dataframe(
        main_data,
        model
    )

    # --------------------------------------------------------
    # FILTERS
    # --------------------------------------------------------

    col1, col2, col3 = (
        st.columns(3)
    )

    if "Contract" in scored.columns:

        contracts = sorted(
            scored[
                "Contract"
            ]
            .dropna()
            .unique()
            .tolist()
        )

        selected_contracts = (
            col1.multiselect(
                "Contract",
                contracts,
                default=contracts
            )
        )

    else:

        selected_contracts = []

    if "InternetService" in scored.columns:

        services = sorted(
            scored[
                "InternetService"
            ]
            .dropna()
            .unique()
            .tolist()
        )

        selected_services = (
            col2.multiselect(
                "Internet Service",
                services,
                default=services
            )
        )

    else:

        selected_services = []

    risk_options = [
        "LOW",
        "MEDIUM",
        "HIGH"
    ]

    selected_risk = (
        col3.multiselect(
            "Risk Level",
            risk_options,
            default=risk_options
        )
    )

    filtered = scored.copy()

    if selected_contracts:

        filtered = filtered[
            filtered[
                "Contract"
            ].isin(
                selected_contracts
            )
        ]

    if selected_services:

        filtered = filtered[
            filtered[
                "InternetService"
            ].isin(
                selected_services
            )
        ]

    if selected_risk:

        filtered = filtered[
            filtered[
                "Risk_Level"
            ].isin(
                selected_risk
            )
        ]

    # --------------------------------------------------------
    # SUMMARY
    # --------------------------------------------------------

    st.subheader(
        "Filtered Customer Summary"
    )

    col1, col2, col3, col4 = (
        st.columns(4)
    )

    col1.metric(
        "Customers",
        f"{len(filtered):,}"
    )

    col2.metric(
        "HIGH Risk",
        f"{(filtered['Risk_Level'] == 'HIGH').sum():,}"
    )

    col3.metric(
        "Avg. Churn Probability",
        f"{filtered['Churn_Probability'].mean():.1%}"
    )

    col4.metric(
        "Monthly Charges",
        format_currency(
            filtered[
                "MonthlyCharges"
            ].sum()
        )
    )

    # --------------------------------------------------------
    # CUSTOMER TABLE
    # --------------------------------------------------------

    st.subheader(
        "Customer Risk Analysis"
    )

    columns = [
        "customerID",
        "gender",
        "SeniorCitizen",
        "Partner",
        "Dependents",
        "tenure",
        "Contract",
        "InternetService",
        "MonthlyCharges",
        "TotalCharges",
        "Churn_Probability",
        "Predicted_Churn",
        "Risk_Level",
        "Retention_Recommendation"
    ]

    display_columns = [
        column
        for column in columns
        if column in filtered.columns
    ]

    table = filtered[
        display_columns
    ].copy()

    if "Churn_Probability" in table.columns:

        table[
            "Churn_Probability"
        ] = (
            table[
                "Churn_Probability"
            ]
            .map(
                lambda x:
                f"{x:.1%}"
            )
        )

    st.dataframe(
        table,
        use_container_width=True,
        height=500
    )


# ============================================================
# CHURN ANALYSIS
# ============================================================

elif page == "Churn Analysis":

    st.header(
        "Churn Analysis & Churn Drivers"
    )

    scored = score_dataframe(
        main_data,
        model
    )

    # --------------------------------------------------------
    # ACTUAL VS PREDICTED
    # --------------------------------------------------------

    if "Churn" in scored.columns:

        st.subheader(
            "Actual vs Predicted Churn"
        )

        actual_rate = (
            scored[
                "Churn"
            ].mean()
            * 100
        )

        predicted_rate = (
            scored[
                "Predicted_Churn"
            ].mean()
            * 100
        )

        col1, col2 = (
            st.columns(2)
        )

        col1.metric(
            "Actual Historical Churn",
            f"{actual_rate:.1f}%"
        )

        col2.metric(
            "Predicted Churn",
            f"{predicted_rate:.1f}%"
        )

        comparison = pd.Series(
            {
                "Actual Churn":
                    actual_rate,
                "Predicted Churn":
                    predicted_rate
            }
        )

        st.bar_chart(
            comparison
        )

    # --------------------------------------------------------
    # CONTRACT ANALYSIS
    # --------------------------------------------------------

    if "Contract" in scored.columns:

        st.subheader(
            "Churn by Contract Type"
        )

        predicted_contract = (
            scored
            .groupby(
                "Contract"
            )[
                "Predicted_Churn"
            ]
            .mean()
            .mul(100)
        )

        st.write(
            "Predicted churn rate"
        )

        st.bar_chart(
            predicted_contract
        )

        if "Churn" in scored.columns:

            actual_contract = (
                scored
                .groupby(
                    "Contract"
                )[
                    "Churn"
                ]
                .mean()
                .mul(100)
            )

            st.write(
                "Actual historical churn rate"
            )

            st.bar_chart(
                actual_contract
            )

    # --------------------------------------------------------
    # INTERNET SERVICE
    # --------------------------------------------------------

    if "InternetService" in scored.columns:

        st.subheader(
            "Churn by Internet Service"
        )

        predicted_service = (
            scored
            .groupby(
                "InternetService"
            )[
                "Predicted_Churn"
            ]
            .mean()
            .mul(100)
        )

        st.write(
            "Predicted churn rate"
        )

        st.bar_chart(
            predicted_service
        )

        if "Churn" in scored.columns:

            actual_service = (
                scored
                .groupby(
                    "InternetService"
                )[
                    "Churn"
                ]
                .mean()
                .mul(100)
            )

            st.write(
                "Actual historical churn rate"
            )

            st.bar_chart(
                actual_service
            )

    # --------------------------------------------------------
    # PAYMENT METHOD
    # --------------------------------------------------------

    if "PaymentMethod" in scored.columns:

        st.subheader(
            "Churn by Payment Method"
        )

        predicted_payment = (
            scored
            .groupby(
                "PaymentMethod"
            )[
                "Predicted_Churn"
            ]
            .mean()
            .mul(100)
            .sort_values(
                ascending=False
            )
        )

        st.write(
            "Predicted churn rate"
        )

        st.bar_chart(
            predicted_payment
        )

        if "Churn" in scored.columns:

            actual_payment = (
                scored
                .groupby(
                    "PaymentMethod"
                )[
                    "Churn"
                ]
                .mean()
                .mul(100)
                .sort_values(
                    ascending=False
                )
            )

            st.write(
                "Actual historical churn rate"
            )

            st.bar_chart(
                actual_payment
            )

    # --------------------------------------------------------
    # CUSTOMER SEGMENT
    # --------------------------------------------------------

    if "Customer_Segment" in scored.columns:

        st.subheader(
            "Churn by Customer Segment"
        )

        predicted_segment = (
            scored
            .groupby(
                "Customer_Segment"
            )[
                "Predicted_Churn"
            ]
            .mean()
            .mul(100)
        )

        st.write(
            "Predicted churn rate"
        )

        st.bar_chart(
            predicted_segment
        )

        if "Churn" in scored.columns:

            actual_segment = (
                scored
                .groupby(
                    "Customer_Segment"
                )[
                    "Churn"
                ]
                .mean()
                .mul(100)
            )

            st.write(
                "Actual historical churn rate"
            )

            st.bar_chart(
                actual_segment
            )

    # --------------------------------------------------------
    # TENURE VS CHURN
    # --------------------------------------------------------

    if (
        "tenure" in scored.columns
        and "Churn" in scored.columns
    ):

        st.subheader(
            "Churn Rate by Customer Tenure"
        )

        tenure_analysis = (
            scored
            .groupby("tenure")[
                "Churn"
            ]
            .mean()
            .mul(100)
        )

        st.line_chart(
            tenure_analysis
        )

    # --------------------------------------------------------
    # MONTHLY CHARGES VS CHURN
    # --------------------------------------------------------

    if (
        "MonthlyCharges" in scored.columns
        and "Churn" in scored.columns
    ):

        st.subheader(
            "Monthly Charges vs Churn"
        )

        charge_bins = pd.cut(
            scored[
                "MonthlyCharges"
            ],
            bins=10
        )

        charge_churn = (
            scored
            .groupby(
                charge_bins,
                observed=False
            )[
                "Churn"
            ]
            .mean()
            .mul(100)
        )

        charge_churn.index = (
            charge_churn.index
            .astype(str)
        )

        st.bar_chart(
            charge_churn
        )

    # --------------------------------------------------------
    # CHURN DISTRIBUTION
    # --------------------------------------------------------

    st.subheader(
        "Churn Distribution"
    )

    if "Churn" in scored.columns:

        churn_distribution = (
            scored[
                "Churn"
            ]
            .map(
                {
                    0: "No Churn",
                    1: "Churn"
                }
            )
            .value_counts()
        )

        st.bar_chart(
            churn_distribution
        )

    # --------------------------------------------------------
    # PROBABILITY DISTRIBUTION
    # --------------------------------------------------------

    st.subheader(
        "Predicted Churn Probability Distribution"
    )

    probability_bins = pd.cut(
        scored[
            "Churn_Probability"
        ],
        bins=[
            0,
            0.10,
            0.20,
            0.30,
            0.40,
            0.50,
            0.60,
            0.70,
            0.80,
            0.90,
            1.00
        ],
        include_lowest=True
    )

    probability_distribution = (
        probability_bins
        .value_counts()
        .sort_index()
    )

    st.bar_chart(
        probability_distribution
    )


# ============================================================
# RISK & RETENTION
# ============================================================

elif page == "Risk & Retention":

    st.header(
        "Risk & Retention Intelligence"
    )

    scored = score_dataframe(
        main_data,
        model
    )

    high_risk = scored[
        scored[
            "Risk_Level"
        ] == "HIGH"
    ].copy()

    medium_risk = scored[
        scored[
            "Risk_Level"
        ] == "MEDIUM"
    ].copy()

    low_risk = scored[
        scored[
            "Risk_Level"
        ] == "LOW"
    ].copy()

    # --------------------------------------------------------
    # RISK SUMMARY
    # --------------------------------------------------------

    col1, col2, col3 = (
        st.columns(3)
    )

    col1.metric(
        "HIGH Risk",
        f"{len(high_risk):,}"
    )

    col2.metric(
        "MEDIUM Risk",
        f"{len(medium_risk):,}"
    )

    col3.metric(
        "LOW Risk",
        f"{len(low_risk):,}"
    )

    # --------------------------------------------------------
    # REVENUE
    # --------------------------------------------------------

    revenue = calculate_revenue_risk(
        scored
    )

    st.subheader(
        "Revenue at Risk"
    )

    col1, col2, col3 = (
        st.columns(3)
    )

    col1.metric(
        "Monthly",
        format_currency(
            revenue["monthly"]
        )
    )

    col2.metric(
        "Annual",
        format_currency(
            revenue["annual"]
        )
    )

    col3.metric(
        "Probability Weighted",
        format_currency(
            revenue["weighted"]
        )
    )

    # --------------------------------------------------------
    # HIGH RISK CUSTOMERS
    # --------------------------------------------------------

    st.subheader(
        "High-Risk Customers"
    )

    high_columns = [
        "customerID",
        "tenure",
        "Contract",
        "InternetService",
        "MonthlyCharges",
        "Churn_Probability",
        "Risk_Level",
        "Retention_Recommendation"
    ]

    high_columns = [
        column
        for column in high_columns
        if column in high_risk.columns
    ]

    high_table = high_risk[
        high_columns
    ].copy()

    if "Churn_Probability" in high_table.columns:

        high_table[
            "Churn_Probability"
        ] = (
            high_table[
                "Churn_Probability"
            ]
            .map(
                lambda x:
                f"{x:.1%}"
            )
        )

    st.dataframe(
        high_table,
        use_container_width=True,
        height=500
    )

    # --------------------------------------------------------
    # RECOMMENDATION SUMMARY
    # --------------------------------------------------------

    st.subheader(
        "Retention Recommendation Summary"
    )

    if not high_risk.empty:

        recommendation_counts = (
            high_risk[
                "Retention_Recommendation"
            ]
            .value_counts()
            .head(10)
        )

        st.bar_chart(
            recommendation_counts
        )

    else:

        st.info(
            "No HIGH-risk customers identified."
        )


# ============================================================
# CSV PREDICTION
# ============================================================

elif page == "CSV Prediction":

    st.header(
        "Customer Churn Prediction from CSV"
    )

    st.write(
        "Upload new customer data to generate "
        "churn probabilities, risk levels, and "
        "retention recommendations."
    )

    st.info(
        "Supported: raw Telco CSV, cleaned CSV, "
        "or feature-engineered CSV."
    )

    st.warning(
        "The trained model is used for inference only. "
        "Uploading a CSV does not retrain the model."
    )

    uploaded_file = st.file_uploader(
        "Upload Customer CSV",
        type=["csv"]
    )

    if uploaded_file is not None:

        uploaded = load_csv(
            uploaded_file
        )

        if uploaded is None:
            st.stop()

        st.success(
            f"CSV loaded: "
            f"{len(uploaded):,} rows × "
            f"{len(uploaded.columns):,} columns"
        )

        st.subheader(
            "Uploaded Data Preview"
        )

        st.dataframe(
            uploaded.head(10),
            use_container_width=True
        )

        # ----------------------------------------------------
        # DETECT FORMAT
        # ----------------------------------------------------

        csv_type = detect_csv_type(
            uploaded
        )

        st.write(
            f"**Detected CSV Type:** "
            f"{csv_type.title()}"
        )

        # ----------------------------------------------------
        # PROCESS
        # ----------------------------------------------------

        if csv_type == "raw":

            prediction_data = (
                clean_uploaded_data(
                    uploaded.copy()
                )
            )

            prediction_data = (
                create_features(
                    prediction_data
                )
            )

            st.success(
                "Raw CSV cleaned and feature engineered."
            )

        elif csv_type == "cleaned":

            prediction_data = (
                create_features(
                    uploaded.copy()
                )
            )

            st.success(
                "Cleaned CSV feature engineered."
            )

        elif csv_type == "featured":

            prediction_data = (
                uploaded.copy()
            )

            st.success(
                "Feature-engineered CSV ready for prediction."
            )

        else:

            st.error(
                "Unsupported CSV format."
            )

            st.write(
                "The uploaded file must contain "
                "the project's required customer columns."
            )

            st.stop()

        # ----------------------------------------------------
        # PREDICT
        # ----------------------------------------------------

        try:

            predictions = score_dataframe(
                prediction_data,
                model
            )

        except Exception as error:

            st.error(
                "Prediction failed."
            )

            st.exception(
                error
            )

            st.stop()

        st.success(
            "Prediction completed successfully."
        )

        # ----------------------------------------------------
        # SUMMARY
        # ----------------------------------------------------

        total = len(
            predictions
        )

        predicted_churn = (
            predictions[
                "Predicted_Churn"
            ].sum()
        )

        high = (
            predictions[
                "Risk_Level"
            ]
            .eq("HIGH")
            .sum()
        )

        medium = (
            predictions[
                "Risk_Level"
            ]
            .eq("MEDIUM")
            .sum()
        )

        average_probability = (
            predictions[
                "Churn_Probability"
            ].mean()
            * 100
        )

        col1, col2, col3, col4, col5 = (
            st.columns(5)
        )

        col1.metric(
            "Customers",
            f"{total:,}"
        )

        col2.metric(
            "Predicted Churn",
            f"{predicted_churn:,}"
        )

        col3.metric(
            "HIGH Risk",
            f"{high:,}"
        )

        col4.metric(
            "MEDIUM Risk",
            f"{medium:,}"
        )

        col5.metric(
            "Average Probability",
            f"{average_probability:.1f}%"
        )

        # ----------------------------------------------------
        # HIGH RISK
        # ----------------------------------------------------

        st.subheader(
            "High-Risk Customers"
        )

        high_risk = predictions[
            predictions[
                "Risk_Level"
            ] == "HIGH"
        ].copy()

        high_columns = [
            "customerID",
            "tenure",
            "Contract",
            "InternetService",
            "MonthlyCharges",
            "Churn_Probability",
            "Risk_Level",
            "Retention_Recommendation"
        ]

        high_columns = [
            column
            for column in high_columns
            if column in high_risk.columns
        ]

        high_display = high_risk[
            high_columns
        ].copy()

        if "Churn_Probability" in high_display.columns:

            high_display[
                "Churn_Probability"
            ] = (
                high_display[
                    "Churn_Probability"
                ]
                .map(
                    lambda x:
                    f"{x:.1%}"
                )
            )

        st.dataframe(
            high_display,
            use_container_width=True,
            height=450
        )

        # ----------------------------------------------------
        # COMPLETE RESULTS
        # ----------------------------------------------------

        st.subheader(
            "Complete Prediction Results"
        )

        result_columns = [
            "customerID",
            "tenure",
            "Contract",
            "InternetService",
            "MonthlyCharges",
            "Churn_Probability",
            "Predicted_Churn",
            "Risk_Level",
            "Retention_Recommendation"
        ]

        result_columns = [
            column
            for column in result_columns
            if column in predictions.columns
        ]

        complete = predictions[
            result_columns
        ].copy()

        if "Churn_Probability" in complete.columns:

            complete[
                "Churn_Probability"
            ] = (
                complete[
                    "Churn_Probability"
                ]
                .map(
                    lambda x:
                    f"{x:.1%}"
                )
            )

        st.dataframe(
            complete,
            use_container_width=True,
            height=500
        )

        # ----------------------------------------------------
        # DOWNLOAD
        # ----------------------------------------------------

        csv_data = (
            predictions
            .to_csv(
                index=False
            )
            .encode("utf-8")
        )

        st.download_button(
            "⬇️ Download Prediction Results",
            data=csv_data,
            file_name="customer_churn_predictions.csv",
            mime="text/csv"
        )

        # ----------------------------------------------------
        # REVENUE RISK
        # ----------------------------------------------------

        st.subheader(
            "Revenue Risk"
        )

        revenue = calculate_revenue_risk(
            predictions
        )

        col1, col2, col3 = (
            st.columns(3)
        )

        col1.metric(
            "HIGH-Risk Monthly Revenue",
            format_currency(
                revenue["monthly"]
            )
        )

        col2.metric(
            "HIGH-Risk Annual Revenue",
            format_currency(
                revenue["annual"]
            )
        )

        col3.metric(
            "Probability-Weighted Annual Risk",
            format_currency(
                revenue["weighted"]
            )
        )


# ============================================================
# INDIVIDUAL CUSTOMER PREDICTION
# ============================================================

elif page == "Individual Customer Prediction":

    st.header(
        "Individual Customer Prediction"
    )

    if "customerID" not in main_data.columns:

        st.error(
            "customerID column is required."
        )

        st.stop()

    customer_ids = (
        main_data[
            "customerID"
        ]
        .dropna()
        .astype(str)
        .tolist()
    )

    selected_customer = st.selectbox(
        "Select Customer",
        customer_ids
    )

    selected = main_data[
        main_data[
            "customerID"
        ].astype(str)
        == selected_customer
    ].iloc[0].copy()

    customer_df = pd.DataFrame(
        [selected]
    )

    # Make sure the selected record
    # has the required engineered features.

    if not set(
        ENGINEERED_COLUMNS
    ).issubset(
        customer_df.columns
    ):

        customer_df = clean_uploaded_data(
            customer_df
        )

        customer_df = create_features(
            customer_df
        )

    try:

        result = score_dataframe(
            customer_df,
            model
        )

    except Exception as error:

        st.error(
            f"Prediction failed: {error}"
        )

        st.stop()

    probability = float(
        result[
            "Churn_Probability"
        ].iloc[0]
    )

    predicted = int(
        result[
            "Predicted_Churn"
        ].iloc[0]
    )

    risk = result[
        "Risk_Level"
    ].iloc[0]

    recommendation = result[
        "Retention_Recommendation"
    ].iloc[0]

    # --------------------------------------------------------
    # DETAILS
    # --------------------------------------------------------

    st.subheader(
        "Customer Information"
    )

    detail_columns = [
        "gender",
        "SeniorCitizen",
        "Partner",
        "Dependents",
        "tenure",
        "Contract",
        "InternetService",
        "MonthlyCharges",
        "TotalCharges",
        "PaymentMethod"
    ]

    details = []

    for column in detail_columns:

        if column in customer_df.columns:

            details.append(
                {
                    "Attribute":
                        column,
                    "Value":
                        customer_df[
                            column
                        ].iloc[0]
                }
            )

    if details:

        st.dataframe(
            pd.DataFrame(details),
            use_container_width=True,
            hide_index=True
        )

    # --------------------------------------------------------
    # PREDICTION
    # --------------------------------------------------------

    st.subheader(
        "Churn Prediction"
    )

    col1, col2, col3 = (
        st.columns(3)
    )

    col1.metric(
        "Churn Probability",
        f"{probability:.1%}"
    )

    col2.metric(
        "Risk Level",
        risk
    )

    col3.metric(
        "Predicted Churn",
        "Yes" if predicted == 1 else "No"
    )

    st.progress(
        min(
            int(
                probability * 100
            ),
            100
        )
    )

    # --------------------------------------------------------
    # RISK FACTORS
    # --------------------------------------------------------

    st.subheader(
        "Why Is This Customer At Risk?"
    )

    factors = identify_risk_factors(
        customer_df.iloc[0]
    )

    for factor in factors:

        st.write(
            f"• {factor}"
        )

    # --------------------------------------------------------
    # RECOMMENDATION
    # --------------------------------------------------------

    st.subheader(
        "Recommended Retention Action"
    )

    st.info(
        recommendation
    )


# ============================================================
# MODEL INSIGHTS
# ============================================================

elif page == "Model Insights":

    st.header(
        "Model Performance & Explainability"
    )

    # --------------------------------------------------------
    # EVALUATION METRICS
    # --------------------------------------------------------

    st.subheader(
        "Selected Model Evaluation"
    )

    if os.path.exists(
        EVALUATION_SUMMARY_PATH
    ):

        try:

            evaluation = pd.read_csv(
                EVALUATION_SUMMARY_PATH
            )

            if "Model" in evaluation.columns:

                matches = evaluation[
                    evaluation[
                        "Model"
                    ]
                    .astype(str)
                    .str.contains(
                        "Gradient",
                        case=False,
                        na=False
                    )
                ]

                if not matches.empty:

                    row = matches.iloc[0]

                    metrics = [
                        "Accuracy",
                        "Precision",
                        "Recall",
                        "F1",
                        "ROC-AUC",
                        "PR-AUC"
                    ]

                    available = [
                        metric
                        for metric in metrics
                        if metric
                        in evaluation.columns
                    ]

                    metric_columns = st.columns(
                        len(available)
                    )

                    for i, metric in enumerate(
                        available
                    ):

                        try:

                            value = float(
                                row[metric]
                            )

                            metric_columns[
                                i
                            ].metric(
                                metric,
                                f"{value:.2%}"
                            )

                        except Exception:

                            metric_columns[
                                i
                            ].metric(
                                metric,
                                str(row[metric])
                            )

                else:

                    st.dataframe(
                        evaluation,
                        use_container_width=True,
                        hide_index=True
                    )

            else:

                st.dataframe(
                    evaluation,
                    use_container_width=True,
                    hide_index=True
                )

        except Exception as error:

            st.warning(
                f"Unable to read evaluation summary: {error}"
            )

    else:

        st.info(
            "Evaluation summary CSV was not found."
        )

    # --------------------------------------------------------
    # MODEL COMPARISON
    # --------------------------------------------------------

    if os.path.exists(
        MODEL_COMPARISON_PATH
    ):

        st.subheader(
            "Model Comparison"
        )

        try:

            comparison = pd.read_csv(
                MODEL_COMPARISON_PATH
            )

            st.dataframe(
                comparison,
                use_container_width=True,
                hide_index=True
            )

        except Exception as error:

            st.warning(
                f"Unable to load model comparison: {error}"
            )

    # --------------------------------------------------------
    # EVALUATION PLOTS
    # --------------------------------------------------------

    if os.path.exists(
        CONFUSION_MATRIX_PATH
    ):

        st.subheader(
            "Confusion Matrix"
        )

        st.image(
            CONFUSION_MATRIX_PATH,
            use_container_width=True
        )

    if os.path.exists(
        ROC_CURVE_PATH
    ):

        st.subheader(
            "ROC Curve"
        )

        st.image(
            ROC_CURVE_PATH,
            use_container_width=True
        )

    if os.path.exists(
        PR_CURVE_PATH
    ):

        st.subheader(
            "Precision-Recall Curve"
        )

        st.image(
            PR_CURVE_PATH,
            use_container_width=True
        )

    if os.path.exists(
        THRESHOLD_ANALYSIS_PATH
    ):

        st.subheader(
            "Threshold Analysis"
        )

        st.image(
            THRESHOLD_ANALYSIS_PATH,
            use_container_width=True
        )

    # --------------------------------------------------------
    # FEATURE IMPORTANCE
    # --------------------------------------------------------

    if os.path.exists(
        FEATURE_IMPORTANCE_PATH
    ):

        st.subheader(
            "Feature Importance"
        )

        try:

            feature_df = pd.read_csv(
                FEATURE_IMPORTANCE_PATH
            )

            st.dataframe(
                feature_df.head(20),
                use_container_width=True,
                hide_index=True
            )

            numeric = (
                feature_df
                .select_dtypes(
                    include=np.number
                )
                .columns
            )

            if len(numeric) > 0:

                importance_column = (
                    numeric[-1]
                )

                chart_data = (
                    feature_df
                    .sort_values(
                        importance_column,
                        ascending=False
                    )
                    .head(15)
                )

                chart_data = (
                    chart_data
                    .set_index(
                        feature_df.columns[0]
                    )
                )

                st.bar_chart(
                    chart_data[
                        importance_column
                    ]
                )

        except Exception as error:

            st.warning(
                f"Unable to load feature importance: {error}"
            )

    # --------------------------------------------------------
    # PERMUTATION IMPORTANCE
    # --------------------------------------------------------

    if os.path.exists(
        PERMUTATION_IMPORTANCE_PATH
    ):

        st.subheader(
            "Permutation Feature Importance"
        )

        try:

            permutation = pd.read_csv(
                PERMUTATION_IMPORTANCE_PATH
            )

            st.dataframe(
                permutation.head(20),
                use_container_width=True,
                hide_index=True
            )

            numeric = (
                permutation
                .select_dtypes(
                    include=np.number
                )
                .columns
            )

            if len(numeric) > 0:

                importance_column = (
                    numeric[-1]
                )

                chart_data = (
                    permutation
                    .sort_values(
                        importance_column,
                        ascending=False
                    )
                    .head(15)
                )

                chart_data = (
                    chart_data
                    .set_index(
                        permutation.columns[0]
                    )
                )

                st.bar_chart(
                    chart_data[
                        importance_column
                    ]
                )

        except Exception as error:

            st.warning(
                f"Unable to load permutation importance: {error}"
            )

    # --------------------------------------------------------
    # MODEL DESCRIPTION
    # --------------------------------------------------------

    st.subheader(
        "Model Configuration"
    )

    st.write(
        f"**Selected Model:** {MODEL_NAME}"
    )

    st.write(
        f"**Churn Threshold:** "
        f"{CHURN_THRESHOLD:.0%}"
    )

    st.write(
        f"**High-Risk Threshold:** "
        f"{HIGH_RISK_THRESHOLD:.0%}"
    )

    st.info(
        "The dashboard performs inference using "
        "the previously trained model. It does not "
        "retrain the model when users upload data."
    )

    # --------------------------------------------------------
    # DATASET LIMITATIONS
    # --------------------------------------------------------

    st.subheader(
        "Dataset Feature Availability"
    )

    st.write(
        "The selected Telco Customer Churn dataset "
        "supports customer tenure, services, charges, "
        "contract, payment method and other customer "
        "attributes."
    )

    st.write(
        "The dataset does not contain historical "
        "support-call counts, payment-delay history, "
        "or month-to-month usage-change measurements. "
        "Therefore, these variables are not artificially "
        "created or fabricated."
    )


# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.caption(
    "Customer Churn Prediction & Business "
    "Intelligence System | Python • Pandas • "
    "Scikit-learn • Streamlit"
)