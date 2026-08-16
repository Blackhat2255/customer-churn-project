# ============================================================
# Customer Churn Prediction & Business Intelligence System
# Module: Model Evaluation
# ============================================================

import os
import warnings

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
    precision_recall_curve
)

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

EVAL_DIR = os.path.join(
    REPORT_DIR,
    "evaluation"
)

os.makedirs(EVAL_DIR, exist_ok=True)


# ------------------------------------------------------------
# Display Helper
# ------------------------------------------------------------

def print_section(title):

    print()
    print("=" * 70)
    print(title)
    print("=" * 70)


# ------------------------------------------------------------
# Load Dataset
# ------------------------------------------------------------

print_section("CUSTOMER CHURN MODEL EVALUATION")

print("Loading feature-engineered dataset...")

df = pd.read_csv(DATA_PATH)

print("Dataset loaded successfully.")
print(f"Rows: {df.shape[0]}")
print(f"Columns: {df.shape[1]}")


# ------------------------------------------------------------
# Prepare Features and Target
# ------------------------------------------------------------

X = df.drop(
    columns=["Churn", "customerID"]
)

y = df["Churn"]


# ------------------------------------------------------------
# Recreate Same Train/Test Split
# ------------------------------------------------------------

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print()
print(f"Training samples: {len(X_train)}")
print(f"Testing samples:  {len(X_test)}")


# ------------------------------------------------------------
# Load Trained Model
# ------------------------------------------------------------

print_section("LOADING BEST MODEL")

print("Loading saved model...")

model = joblib.load(MODEL_PATH)

print("Model loaded successfully.")
print("Model type: Gradient Boosting pipeline")


# ------------------------------------------------------------
# Generate Predictions
# ------------------------------------------------------------

print_section("GENERATING TEST PREDICTIONS")

y_pred = model.predict(X_test)

y_probability = model.predict_proba(X_test)[:, 1]

print("Predictions generated successfully.")


# ------------------------------------------------------------
# Basic Evaluation Metrics
# ------------------------------------------------------------

accuracy = accuracy_score(
    y_test,
    y_pred
)

precision = precision_score(
    y_test,
    y_pred,
    zero_division=0
)

recall = recall_score(
    y_test,
    y_pred,
    zero_division=0
)

f1 = f1_score(
    y_test,
    y_pred,
    zero_division=0
)

roc_auc = roc_auc_score(
    y_test,
    y_probability
)

pr_auc = average_precision_score(
    y_test,
    y_probability
)


print_section("MODEL PERFORMANCE")

print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1 Score : {f1:.4f}")
print(f"ROC-AUC  : {roc_auc:.4f}")
print(f"PR-AUC   : {pr_auc:.4f}")


# ------------------------------------------------------------
# Classification Report
# ------------------------------------------------------------

print_section("CLASSIFICATION REPORT")

classification_report_text = classification_report(
    y_test,
    y_pred,
    target_names=[
        "Retained",
        "Churned"
    ],
    zero_division=0
)

print(classification_report_text)

classification_report_path = os.path.join(
    EVAL_DIR,
    "classification_report.txt"
)

with open(
    classification_report_path,
    "w",
    encoding="utf-8"
) as file:

    file.write(
        "CUSTOMER CHURN CLASSIFICATION REPORT\n"
    )

    file.write("=" * 50 + "\n\n")

    file.write(
        classification_report_text
    )

    file.write("\n\n")

    file.write(
        f"Accuracy : {accuracy:.4f}\n"
    )

    file.write(
        f"Precision: {precision:.4f}\n"
    )

    file.write(
        f"Recall   : {recall:.4f}\n"
    )

    file.write(
        f"F1 Score : {f1:.4f}\n"
    )

    file.write(
        f"ROC-AUC  : {roc_auc:.4f}\n"
    )

    file.write(
        f"PR-AUC   : {pr_auc:.4f}\n"
    )


# ------------------------------------------------------------
# Confusion Matrix
# ------------------------------------------------------------

print_section("CONFUSION MATRIX")

cm = confusion_matrix(
    y_test,
    y_pred
)

print(cm)

plt.figure(figsize=(7, 5))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=[
        "Predicted Retained",
        "Predicted Churned"
    ],
    yticklabels=[
        "Actual Retained",
        "Actual Churned"
    ]
)

plt.title(
    "Confusion Matrix - Gradient Boosting"
)

plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.tight_layout()

confusion_matrix_path = os.path.join(
    EVAL_DIR,
    "confusion_matrix.png"
)

plt.savefig(
    confusion_matrix_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ------------------------------------------------------------
# ROC Curve
# ------------------------------------------------------------

print_section("ROC CURVE")

fpr, tpr, roc_thresholds = roc_curve(
    y_test,
    y_probability
)

plt.figure(figsize=(8, 6))

plt.plot(
    fpr,
    tpr,
    linewidth=2,
    label=f"Gradient Boosting (AUC = {roc_auc:.4f})"
)

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    linewidth=1
)

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")

plt.title(
    "ROC Curve - Customer Churn Prediction"
)

plt.legend(
    loc="lower right"
)

plt.grid(
    alpha=0.3
)

plt.tight_layout()

roc_curve_path = os.path.join(
    EVAL_DIR,
    "roc_curve.png"
)

plt.savefig(
    roc_curve_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ------------------------------------------------------------
# Precision-Recall Curve
# ------------------------------------------------------------

print_section("PRECISION-RECALL CURVE")

precision_values, recall_values, pr_thresholds = (
    precision_recall_curve(
        y_test,
        y_probability
    )
)

plt.figure(figsize=(8, 6))

plt.plot(
    recall_values,
    precision_values,
    linewidth=2,
    label=f"PR-AUC = {pr_auc:.4f}"
)

plt.xlabel("Recall")
plt.ylabel("Precision")

plt.title(
    "Precision-Recall Curve - Customer Churn Prediction"
)

plt.legend(
    loc="lower left"
)

plt.grid(
    alpha=0.3
)

plt.tight_layout()

pr_curve_path = os.path.join(
    EVAL_DIR,
    "precision_recall_curve.png"
)

plt.savefig(
    pr_curve_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ------------------------------------------------------------
# Threshold Analysis
# ------------------------------------------------------------

print_section("THRESHOLD ANALYSIS")

thresholds = np.arange(
    0.10,
    0.71,
    0.05
)

threshold_results = []

for threshold in thresholds:

    threshold_predictions = (
        y_probability >= threshold
    ).astype(int)

    threshold_accuracy = accuracy_score(
        y_test,
        threshold_predictions
    )

    threshold_precision = precision_score(
        y_test,
        threshold_predictions,
        zero_division=0
    )

    threshold_recall = recall_score(
        y_test,
        threshold_predictions,
        zero_division=0
    )

    threshold_f1 = f1_score(
        y_test,
        threshold_predictions,
        zero_division=0
    )

    threshold_results.append(
        {
            "Threshold": threshold,
            "Accuracy": threshold_accuracy,
            "Precision": threshold_precision,
            "Recall": threshold_recall,
            "F1": threshold_f1
        }
    )


threshold_df = pd.DataFrame(
    threshold_results
)

print(
    threshold_df.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)


# ------------------------------------------------------------
# Find Best F1 Threshold
# ------------------------------------------------------------

best_threshold_row = threshold_df.loc[
    threshold_df["F1"].idxmax()
]

best_threshold = float(
    best_threshold_row["Threshold"]
)

print()
print(
    f"Best threshold based on F1-score: "
    f"{best_threshold:.2f}"
)

print(
    f"Accuracy : "
    f"{best_threshold_row['Accuracy']:.4f}"
)

print(
    f"Precision: "
    f"{best_threshold_row['Precision']:.4f}"
)

print(
    f"Recall   : "
    f"{best_threshold_row['Recall']:.4f}"
)

print(
    f"F1 Score : "
    f"{best_threshold_row['F1']:.4f}"
)


# ------------------------------------------------------------
# Save Threshold Results
# ------------------------------------------------------------

threshold_results_path = os.path.join(
    REPORT_DIR,
    "threshold_tuning_results.csv"
)

threshold_df.to_csv(
    threshold_results_path,
    index=False
)

print()
print(
    "Threshold results saved to:"
)

print(
    threshold_results_path
)


# ------------------------------------------------------------
# Threshold Performance Plot
# ------------------------------------------------------------

plt.figure(figsize=(9, 6))

plt.plot(
    threshold_df["Threshold"],
    threshold_df["Precision"],
    marker="o",
    label="Precision"
)

plt.plot(
    threshold_df["Threshold"],
    threshold_df["Recall"],
    marker="o",
    label="Recall"
)

plt.plot(
    threshold_df["Threshold"],
    threshold_df["F1"],
    marker="o",
    linewidth=2,
    label="F1 Score"
)

plt.axvline(
    best_threshold,
    linestyle="--",
    linewidth=1,
    label=f"Best F1 Threshold = {best_threshold:.2f}"
)

plt.xlabel("Probability Threshold")
plt.ylabel("Score")

plt.title(
    "Threshold Analysis - Churn Prediction"
)

plt.legend()

plt.grid(
    alpha=0.3
)

plt.tight_layout()

threshold_plot_path = os.path.join(
    EVAL_DIR,
    "threshold_analysis.png"
)

plt.savefig(
    threshold_plot_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ------------------------------------------------------------
# Apply Best Threshold
# ------------------------------------------------------------

print_section("BEST THRESHOLD EVALUATION")

y_threshold_pred = (
    y_probability >= best_threshold
).astype(int)

threshold_accuracy = accuracy_score(
    y_test,
    y_threshold_pred
)

threshold_precision = precision_score(
    y_test,
    y_threshold_pred,
    zero_division=0
)

threshold_recall = recall_score(
    y_test,
    y_threshold_pred,
    zero_division=0
)

threshold_f1 = f1_score(
    y_test,
    y_threshold_pred,
    zero_division=0
)

print(
    f"Selected threshold: {best_threshold:.2f}"
)

print(
    f"Accuracy : {threshold_accuracy:.4f}"
)

print(
    f"Precision: {threshold_precision:.4f}"
)

print(
    f"Recall   : {threshold_recall:.4f}"
)

print(
    f"F1 Score : {threshold_f1:.4f}"
)


# ------------------------------------------------------------
# Save Test Predictions
# ------------------------------------------------------------

prediction_results = X_test.copy()

prediction_results["Actual_Churn"] = y_test.values

prediction_results["Churn_Probability"] = (
    y_probability
)

prediction_results["Default_Prediction"] = (
    y_pred
)

prediction_results["Threshold_Prediction"] = (
    y_threshold_pred
)

prediction_results_path = os.path.join(
    REPORT_DIR,
    "test_predictions.csv"
)

prediction_results.to_csv(
    prediction_results_path,
    index=False
)


# ------------------------------------------------------------
# Evaluation Summary
# ------------------------------------------------------------

evaluation_summary = pd.DataFrame(
    [
        {
            "Metric": "Accuracy",
            "Default_Threshold": accuracy,
            "Best_Threshold": threshold_accuracy
        },
        {
            "Metric": "Precision",
            "Default_Threshold": precision,
            "Best_Threshold": threshold_precision
        },
        {
            "Metric": "Recall",
            "Default_Threshold": recall,
            "Best_Threshold": threshold_recall
        },
        {
            "Metric": "F1",
            "Default_Threshold": f1,
            "Best_Threshold": threshold_f1
        },
        {
            "Metric": "ROC-AUC",
            "Default_Threshold": roc_auc,
            "Best_Threshold": roc_auc
        },
        {
            "Metric": "PR-AUC",
            "Default_Threshold": pr_auc,
            "Best_Threshold": pr_auc
        }
    ]
)

summary_path = os.path.join(
    EVAL_DIR,
    "evaluation_summary.csv"
)

evaluation_summary.to_csv(
    summary_path,
    index=False
)


# ------------------------------------------------------------
# Final Output
# ------------------------------------------------------------

print_section("MODEL EVALUATION COMPLETE")

print("Evaluation files saved to:")
print(EVAL_DIR)

print()
print("Generated files:")

print("- classification_report.txt")
print("- confusion_matrix.png")
print("- roc_curve.png")
print("- precision_recall_curve.png")
print("- threshold_analysis.png")
print("- evaluation_summary.csv")

print()
print("Additional report:")
print("- threshold_tuning_results.csv")
print("- test_predictions.csv")

print()
print("Model evaluation completed successfully.")