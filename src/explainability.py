# ============================================================
# Customer Churn Prediction & Business Intelligence System
# Explainability and Feature Importance
# ============================================================

import os

# Use a non-interactive backend to avoid Tkinter problems
import matplotlib
matplotlib.use("Agg")

import joblib
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.inspection import permutation_importance


# ------------------------------------------------------------
# Project Paths
# ------------------------------------------------------------

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
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

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "reports",
    "explainability"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ------------------------------------------------------------
# Load Dataset
# ------------------------------------------------------------

print("=" * 70)
print("MODEL EXPLAINABILITY AND FEATURE IMPORTANCE")
print("=" * 70)

print("\nLoading feature-engineered dataset...")

df = pd.read_csv(DATA_PATH)

print("Dataset loaded successfully.")
print(f"Rows: {df.shape[0]}")
print(f"Columns: {df.shape[1]}")


# ------------------------------------------------------------
# Load Trained Model
# ------------------------------------------------------------

print("\nLoading trained churn model...")

model = joblib.load(MODEL_PATH)

print("Model loaded successfully.")
print(f"Model type: {type(model).__name__}")


# ------------------------------------------------------------
# Display Pipeline Steps
# ------------------------------------------------------------

print("\nModel pipeline steps:")

if hasattr(model, "named_steps"):

    for step_name, step_object in model.named_steps.items():

        print(
            f"- {step_name}: "
            f"{type(step_object).__name__}"
        )


# ------------------------------------------------------------
# Identify Final Estimator
# ------------------------------------------------------------

if hasattr(model, "steps"):

    classifier = model.steps[-1][1]

else:

    classifier = model


print(
    f"\nFinal estimator detected: "
    f"{type(classifier).__name__}"
)


# ------------------------------------------------------------
# Identify Preprocessor
# ------------------------------------------------------------

preprocessor = None

if hasattr(model, "named_steps"):

    if "preprocessor" in model.named_steps:

        preprocessor = model.named_steps[
            "preprocessor"
        ]

        print(
            "Preprocessor detected: "
            "preprocessor"
        )


# ------------------------------------------------------------
# Prepare Features and Target
# ------------------------------------------------------------

X = df.drop(
    columns=[
        "Churn",
        "customerID"
    ]
)

y = df["Churn"]


# ============================================================
# PART 1 — MODEL FEATURE IMPORTANCE
# ============================================================

print("\n" + "=" * 70)
print("MODEL FEATURE IMPORTANCE")
print("=" * 70)


if hasattr(
    classifier,
    "feature_importances_"
):

    importance_values = (
        classifier.feature_importances_
    )


    # Get transformed feature names
    if preprocessor is not None:

        feature_names = (
            preprocessor
            .get_feature_names_out()
        )

    else:

        feature_names = X.columns


    # Safety check
    if len(feature_names) != len(
        importance_values
    ):

        print(
            "\nWarning: feature name count "
            "does not match importance count."
        )

        feature_names = [
            f"Feature_{i}"
            for i in range(
                len(importance_values)
            )
        ]


    feature_importance = pd.DataFrame({

        "Feature": feature_names,

        "Importance": importance_values

    })


    feature_importance = (
        feature_importance
        .sort_values(
            by="Importance",
            ascending=False
        )
        .reset_index(drop=True)
    )


else:

    print(
        "\nThe selected model does not provide "
        "feature_importances_."
    )

    feature_importance = pd.DataFrame()


# ------------------------------------------------------------
# Display Top Model Features
# ------------------------------------------------------------

if not feature_importance.empty:

    print("\nTop 20 model features:")

    print(
        feature_importance
        .head(20)
        .to_string(index=False)
    )


# ------------------------------------------------------------
# Save Detailed Feature Importance
# ------------------------------------------------------------

feature_importance_path = os.path.join(
    OUTPUT_DIR,
    "model_feature_importance.csv"
)

feature_importance.to_csv(
    feature_importance_path,
    index=False
)

print(
    f"\nDetailed feature importance saved to:\n"
    f"{feature_importance_path}"
)


# ============================================================
# PART 2 — AGGREGATE ONE-HOT FEATURES
# ============================================================

print("\n" + "=" * 70)
print("AGGREGATED FEATURE IMPORTANCE")
print("=" * 70)


if not feature_importance.empty:

    def get_original_feature(
        feature_name
    ):

        feature_name = str(
            feature_name
        )

        # Remove transformer prefixes
        feature_name = feature_name.replace(
            "numeric__",
            ""
        )

        feature_name = feature_name.replace(
            "categorical__",
            ""
        )

        # Map one-hot encoded categories
        categorical_columns = [
            "gender",
            "Partner",
            "Dependents",
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
            "Customer_Segment"
        ]

        for column in categorical_columns:

            if feature_name.startswith(
                column + "_"
            ):

                return column


        # Numeric / engineered features
        return feature_name


    feature_importance[
        "Original_Feature"
    ] = (
        feature_importance[
            "Feature"
        ]
        .apply(
            get_original_feature
        )
    )


    aggregated_importance = (
        feature_importance
        .groupby(
            "Original_Feature",
            as_index=False
        )["Importance"]
        .sum()
        .sort_values(
            by="Importance",
            ascending=False
        )
        .reset_index(drop=True)
    )


else:

    aggregated_importance = (
        pd.DataFrame()
    )


# ------------------------------------------------------------
# Display Aggregated Features
# ------------------------------------------------------------

if not aggregated_importance.empty:

    print(
        "\nTop original features:"
    )

    print(
        aggregated_importance
        .head(20)
        .to_string(index=False)
    )


# ------------------------------------------------------------
# Save Aggregated Importance
# ------------------------------------------------------------

aggregated_path = os.path.join(
    OUTPUT_DIR,
    "aggregated_feature_importance.csv"
)

aggregated_importance.to_csv(
    aggregated_path,
    index=False
)

print(
    f"\nAggregated feature importance saved to:\n"
    f"{aggregated_path}"
)


# ============================================================
# PART 3 — FEATURE IMPORTANCE VISUALIZATION
# ============================================================

if not aggregated_importance.empty:

    top_features = (
        aggregated_importance
        .head(15)
        .sort_values(
            by="Importance"
        )
    )


    plt.figure(
        figsize=(10, 7)
    )

    plt.barh(
        top_features[
            "Original_Feature"
        ],
        top_features[
            "Importance"
        ]
    )

    plt.xlabel(
        "Importance"
    )

    plt.ylabel(
        "Feature"
    )

    plt.title(
        "Top Features Influencing Customer Churn"
    )

    plt.tight_layout()


    importance_plot_path = os.path.join(
        OUTPUT_DIR,
        "feature_importance.png"
    )

    plt.savefig(
        importance_plot_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(
        f"\nFeature importance chart saved to:\n"
        f"{importance_plot_path}"
    )


# ============================================================
# PART 4 — PERMUTATION IMPORTANCE
# ============================================================

print("\n" + "=" * 70)
print("PERMUTATION IMPORTANCE")
print("=" * 70)

print(
    "\nCalculating permutation importance..."
)

# Use a representative sample
sample_size = min(
    1500,
    len(X)
)

X_sample = X.sample(
    n=sample_size,
    random_state=42
)

y_sample = y.loc[
    X_sample.index
]


# IMPORTANT:
# n_jobs=1 avoids Windows Tkinter/threading
# issues with Python 3.14.
permutation = permutation_importance(

    model,

    X_sample,

    y_sample,

    scoring="roc_auc",

    n_repeats=10,

    random_state=42,

    n_jobs=1
)


permutation_importance_df = pd.DataFrame({

    "Feature": X_sample.columns,

    "Importance_Mean":
        permutation.importances_mean,

    "Importance_STD":
        permutation.importances_std

})


permutation_importance_df = (
    permutation_importance_df
    .sort_values(
        by="Importance_Mean",
        ascending=False
    )
    .reset_index(drop=True)
)


# ------------------------------------------------------------
# Display Permutation Results
# ------------------------------------------------------------

print(
    "\nTop permutation-important features:"
)

print(
    permutation_importance_df
    .head(20)
    .to_string(index=False)
)


# ------------------------------------------------------------
# Save Permutation Importance
# ------------------------------------------------------------

permutation_path = os.path.join(
    OUTPUT_DIR,
    "permutation_feature_importance.csv"
)

permutation_importance_df.to_csv(
    permutation_path,
    index=False
)

print(
    f"\nPermutation importance saved to:\n"
    f"{permutation_path}"
)


# ============================================================
# PART 5 — PERMUTATION IMPORTANCE VISUALIZATION
# ============================================================

top_permutation = (
    permutation_importance_df
    .head(15)
    .sort_values(
        by="Importance_Mean"
    )
)


plt.figure(
    figsize=(10, 7)
)

plt.barh(
    top_permutation[
        "Feature"
    ],
    top_permutation[
        "Importance_Mean"
    ]
)

plt.xlabel(
    "Mean Decrease in ROC-AUC"
)

plt.ylabel(
    "Feature"
)

plt.title(
    "Permutation Feature Importance"
)

plt.tight_layout()


permutation_plot_path = os.path.join(
    OUTPUT_DIR,
    "permutation_importance.png"
)

plt.savefig(
    permutation_plot_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print(
    f"Permutation importance chart saved to:\n"
    f"{permutation_plot_path}"
)


# ============================================================
# PART 6 — BUSINESS INTERPRETATION
# ============================================================

print("\n" + "=" * 70)
print("BUSINESS INTERPRETATION")
print("=" * 70)

print("""
The feature importance results identify customer characteristics
that are most useful to the trained model when distinguishing
between customers who churn and customers who remain.

These features should be interpreted as predictive indicators,
not as proof that a feature directly causes customer churn.

Permutation importance provides an additional model-agnostic
perspective by measuring how model ROC-AUC changes when individual
features are randomly shuffled.
""")


# ============================================================
# FINAL OUTPUT
# ============================================================

print("=" * 70)
print("EXPLAINABILITY ANALYSIS COMPLETE")
print("=" * 70)

print("\nGenerated files:")

print(
    "- model_feature_importance.csv"
)

print(
    "- aggregated_feature_importance.csv"
)

print(
    "- feature_importance.png"
)

print(
    "- permutation_feature_importance.csv"
)

print(
    "- permutation_importance.png"
)

print(
    "\nExplainability analysis completed successfully."
)