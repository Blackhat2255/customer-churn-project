# ============================================================
# Customer Churn Prediction & Business Intelligence System
# Module: Model Training
# ============================================================

import os
import warnings

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

from xgboost import XGBClassifier


# ------------------------------------------------------------
# Configuration
# ------------------------------------------------------------

warnings.filterwarnings("ignore")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_PATH = os.path.join(
    BASE_DIR,
    "data",
    "processed",
    "featured_churn_data.csv"
)

MODEL_DIR = os.path.join(BASE_DIR, "models")
REPORT_DIR = os.path.join(BASE_DIR, "reports")

os.makedirs(MODEL_DIR, exist_ok=True)
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
# Load Dataset
# ------------------------------------------------------------

print_section("CUSTOMER CHURN MODEL TRAINING")

print("Loading feature-engineered dataset...")

df = pd.read_csv(DATA_PATH)

print("Dataset loaded successfully.")
print(f"Rows: {df.shape[0]}")
print(f"Columns: {df.shape[1]}")


# ------------------------------------------------------------
# Prepare Features and Target
# ------------------------------------------------------------

print_section("PREPARING FEATURES AND TARGET")

# Customer ID is an identifier and should not be used
# as a predictive feature.
X = df.drop(columns=["Churn", "customerID"])

y = df["Churn"]

print(f"Feature count: {X.shape[1]}")
print(f"Target count: {y.shape[0]}")

print()
print("Target distribution:")
print(y.value_counts().sort_index())

print()
print("Target distribution percentage:")
print(
    (y.value_counts(normalize=True).sort_index() * 100)
    .round(2)
)


# ------------------------------------------------------------
# Identify Feature Types
# ------------------------------------------------------------

numeric_features = X.select_dtypes(
    include=["int64", "float64"]
).columns.tolist()

categorical_features = X.select_dtypes(
    include=["object", "string"]
).columns.tolist()

print()
print("Numerical features:")
print(numeric_features)

print()
print("Categorical features:")
print(categorical_features)


# ------------------------------------------------------------
# Train/Test Split
# ------------------------------------------------------------

print_section("TRAIN / TEST SPLIT")

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print(f"Training samples: {len(X_train)}")
print(f"Testing samples:  {len(X_test)}")

print()
print("Training target distribution:")
print(y_train.value_counts().sort_index())

print()
print("Testing target distribution:")
print(y_test.value_counts().sort_index())


# ------------------------------------------------------------
# Numerical Preprocessing
# ------------------------------------------------------------

numeric_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="median")
        ),
        (
            "scaler",
            StandardScaler()
        )
    ]
)


# ------------------------------------------------------------
# Categorical Preprocessing
# ------------------------------------------------------------

categorical_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="most_frequent")
        ),
        (
            "encoder",
            OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False
            )
        )
    ]
)


# ------------------------------------------------------------
# Combined Preprocessor
# ------------------------------------------------------------

preprocessor = ColumnTransformer(
    transformers=[
        (
            "numeric",
            numeric_pipeline,
            numeric_features
        ),
        (
            "categorical",
            categorical_pipeline,
            categorical_features
        )
    ]
)


# ------------------------------------------------------------
# Define Models
# ------------------------------------------------------------

models = {

    "Logistic Regression": LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        random_state=42
    ),

    "Decision Tree": DecisionTreeClassifier(
        max_depth=6,
        min_samples_split=10,
        min_samples_leaf=5,
        class_weight="balanced",
        random_state=42
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=300,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    ),

    "Gradient Boosting": GradientBoostingClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=3,
        random_state=42
    ),

    "XGBoost": XGBClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1
    )
}


# ------------------------------------------------------------
# Train and Evaluate Models
# ------------------------------------------------------------

print_section("MODEL TRAINING AND EVALUATION")

results = []
trained_models = {}

for model_name, model in models.items():

    print()
    print("-" * 70)
    print(f"Training: {model_name}")
    print("-" * 70)

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model)
        ]
    )

    # Train
    pipeline.fit(X_train, y_train)

    # Predictions
    y_pred = pipeline.predict(X_test)

    # Probability predictions
    y_probability = pipeline.predict_proba(X_test)[:, 1]

    # Metrics
    accuracy = accuracy_score(y_test, y_pred)

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

    results.append(
        {
            "Model": model_name,
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall,
            "F1": f1,
            "ROC-AUC": roc_auc,
            "PR-AUC": pr_auc
        }
    )

    trained_models[model_name] = pipeline

    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")
    print(f"ROC-AUC  : {roc_auc:.4f}")
    print(f"PR-AUC   : {pr_auc:.4f}")


# ------------------------------------------------------------
# Model Comparison
# ------------------------------------------------------------

print_section("MODEL COMPARISON")

results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    by="ROC-AUC",
    ascending=False
).reset_index(drop=True)

print(
    results_df.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)


# ------------------------------------------------------------
# Select Best Model
# ------------------------------------------------------------

best_model_name = results_df.iloc[0]["Model"]

best_model = trained_models[best_model_name]

print()
print(f"Best model based on ROC-AUC: {best_model_name}")

print()
print("Best model performance:")

best_row = results_df.iloc[0]

print(f"Accuracy : {best_row['Accuracy']:.4f}")
print(f"Precision: {best_row['Precision']:.4f}")
print(f"Recall   : {best_row['Recall']:.4f}")
print(f"F1 Score : {best_row['F1']:.4f}")
print(f"ROC-AUC  : {best_row['ROC-AUC']:.4f}")
print(f"PR-AUC   : {best_row['PR-AUC']:.4f}")


# ------------------------------------------------------------
# Save Model Comparison
# ------------------------------------------------------------

results_path = os.path.join(
    REPORT_DIR,
    "model_comparison_results.csv"
)

results_df.to_csv(
    results_path,
    index=False
)

print()
print("Model comparison saved to:")
print(results_path)


# ------------------------------------------------------------
# Save Best Model
# ------------------------------------------------------------

model_path = os.path.join(
    MODEL_DIR,
    "best_churn_model.pkl"
)

joblib.dump(
    best_model,
    model_path
)

print()
print("Best model saved to:")
print(model_path)


# ------------------------------------------------------------
# Save Test Data for Later Evaluation
# ------------------------------------------------------------

test_data = X_test.copy()

test_data["Actual_Churn"] = y_test.values

test_data_path = os.path.join(
    REPORT_DIR,
    "test_set_predictions_base.csv"
)

test_data.to_csv(
    test_data_path,
    index=False
)

print()
print("Test dataset saved to:")
print(test_data_path)


# ------------------------------------------------------------
# Final Summary
# ------------------------------------------------------------

print_section("MODEL TRAINING COMPLETE")

print(f"Models trained: {len(models)}")
print(f"Best model: {best_model_name}")
print(f"Selection metric: ROC-AUC")
print(f"Best ROC-AUC: {best_row['ROC-AUC']:.4f}")

print()
print("Saved files:")
print(f"- {results_path}")
print(f"- {model_path}")
print(f"- {test_data_path}")

print()
print("Model training completed successfully.")