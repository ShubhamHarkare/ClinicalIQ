"""
model.py — ClinicalIQ Patient Readmission Prediction
Training pipeline with hyperparameter tuning, threshold optimization,
SHAP explainability, and model persistence.
"""

import pandas as pd
import numpy as np
import logging
import time
import joblib

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    confusion_matrix,
    precision_recall_curve
)
import xgboost as xgb
import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# --- LOGGING CONFIGURATION ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

# --- CONFIGURATION ---
PROCESSED_DATA_PATH  = "data/processed/features.csv"
MODEL_SAVE_PATH      = "src/xgboost_readmission.pkl"
FEATURES_SAVE_PATH   = "src/model_features.pkl"
SHAP_PLOT_PATH       = "src/shap_summary.png"


def load_data(path: str) -> pd.DataFrame:
    """
    Load the feature-engineered CSV produced by features.py.

    Args:
        path: Path to the processed features CSV.

    Returns:
        DataFrame with all engineered features and target column.

    Raises:
        FileNotFoundError: If the file does not exist at the given path.
    """
    try:
        df = pd.read_csv(path)
        logger.info(f"Loaded data from {path} — shape: {df.shape}")
        return df
    except FileNotFoundError:
        logger.error(
            f"Could not find {path}. "
            "Ensure features.py has run successfully before training."
        )
        raise


def preprocess(df: pd.DataFrame):
    """
    Clean and encode the feature DataFrame for XGBoost training.

    Steps:
        - Drop non-feature identifier columns
        - Fill numeric nulls with column median (safer than -1 for tree models)
        - Fill categorical nulls with 'Unknown'
        - One-hot encode demographic categorical columns
        - Sanitize column names for XGBoost compatibility

    Args:
        df: Raw feature DataFrame loaded from CSV.

    Returns:
        Tuple of (X, y) where X is the feature matrix and y is the binary target.
    """
    logger.info("Preprocessing data...")

    # Drop identifier columns that should not be features
    drop_cols = [col for col in ["encounter_id"] if col in df.columns]
    if drop_cols:
        df = df.drop(columns=drop_cols)
        logger.info(f"Dropped identifier columns: {drop_cols}")

    # Separate numeric and categorical columns
    numeric_cols     = df.select_dtypes(include=["number"]).columns.tolist()
    categorical_cols = df.select_dtypes(exclude=["number"]).columns.tolist()

    # Remove target from numeric_cols if present
    if "readmitted" in numeric_cols:
        numeric_cols.remove("readmitted")

    # Fill nulls — median for numerics, 'Unknown' for categoricals
    df[numeric_cols]     = df[numeric_cols].fillna(df[numeric_cols].median())
    df[categorical_cols] = df[categorical_cols].fillna("Unknown")

    # One-hot encode demographic columns
    categorical_to_encode = [c for c in ["age", "gender", "race"] if c in df.columns]
    df = pd.get_dummies(df, columns=categorical_to_encode, drop_first=True)

    # Sanitize column names — XGBoost rejects brackets and special chars
    df.columns = df.columns.str.replace(r"[\[\]<>]", "_", regex=True)

    X = df.drop(columns=["readmitted"])
    y = df["readmitted"].astype(int)

    logger.info(f"Feature matrix shape: {X.shape}")
    logger.info(f"Target distribution:\n{y.value_counts(normalize=True).round(3)}")

    return X, y


def tune_and_train(X_train, y_train):
    """
    Run GridSearchCV over an XGBoost classifier with scale_pos_weight
    to handle class imbalance, optimizing for AUC-ROC.

    Args:
        X_train: Training feature matrix.
        y_train: Training target vector.

    Returns:
        Tuple of (best_model, best_params, best_cv_score).
    """
    logger.info("Running GridSearchCV for hyperparameter tuning...")

    # Compute class imbalance ratio for scale_pos_weight
    ratio = float(np.sum(y_train == 0)) / np.sum(y_train == 1)
    logger.info(f"Class imbalance ratio (neg/pos): {ratio:.2f} — applying as scale_pos_weight")

    base_model = xgb.XGBClassifier(
        random_state=42,
        eval_metric="logloss",
        scale_pos_weight=ratio,
        use_label_encoder=False
    )

    # Expanded grid — meaningful coverage without excessive compute
    param_grid = {
        "max_depth":        [3, 5, 7],
        "n_estimators":     [100, 200, 300],
        "learning_rate":    [0.01, 0.05, 0.1],
        "subsample":        [0.8, 1.0],
        "min_child_weight": [1, 5]
    }

    grid_search = GridSearchCV(
        estimator=base_model,
        param_grid=param_grid,
        scoring="roc_auc",
        cv=3,
        verbose=1,
        n_jobs=-1
    )

    grid_search.fit(X_train, y_train)

    best_model    = grid_search.best_estimator_
    best_params   = grid_search.best_params_
    best_cv_score = grid_search.best_score_

    logger.info(f"Best Parameters : {best_params}")
    logger.info(f"Best CV AUC-ROC : {best_cv_score:.4f}")

    return best_model, best_params, best_cv_score


def find_optimal_threshold(y_test, y_proba) -> float:
    """
    Find the classification threshold that maximizes the F1 score
    on the test set. This is especially important for imbalanced
    healthcare datasets where the default 0.5 threshold is suboptimal.

    Args:
        y_test:  True binary labels.
        y_proba: Predicted probabilities for the positive class.

    Returns:
        Optimal threshold value as a float.
    """
    precisions, recalls, thresholds = precision_recall_curve(y_test, y_proba)

    # Avoid divide-by-zero with small epsilon
    f1_scores = (
        2 * (precisions * recalls) /
        (precisions + recalls + 1e-8)
    )

    optimal_idx       = np.argmax(f1_scores)
    optimal_threshold = thresholds[optimal_idx]

    logger.info(f"Optimal threshold (max F1): {optimal_threshold:.4f}")
    logger.info(f"F1 at optimal threshold   : {f1_scores[optimal_idx]:.4f}")

    return optimal_threshold


def evaluate(model, X_test, y_test):
    """
    Evaluate the trained model at both the default (0.5) and
    optimal thresholds, and log all key metrics.

    Args:
        model:  Trained XGBoost classifier.
        X_test: Test feature matrix.
        y_test: True test labels.

    Returns:
        Dictionary with auc, optimal_threshold, and classification reports.
    """
    logger.info("Evaluating model...")

    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    auc     = roc_auc_score(y_test, y_proba)

    logger.info(f"AUC-ROC (default threshold 0.5): {auc:.4f}")

    print("\n--- Confusion Matrix (threshold = 0.5) ---")
    print(confusion_matrix(y_test, y_pred))

    print("\n--- Classification Report (threshold = 0.5) ---")
    print(classification_report(y_test, y_pred))

    # Optimal threshold evaluation
    optimal_threshold = find_optimal_threshold(y_test, y_proba)
    y_pred_optimal    = (y_proba >= optimal_threshold).astype(int)

    print(f"\n--- Classification Report (threshold = {optimal_threshold:.4f}) ---")
    print(classification_report(y_test, y_pred_optimal))

    print("\n--- Confusion Matrix (optimal threshold) ---")
    print(confusion_matrix(y_test, y_pred_optimal))

    return {
        "auc":                auc,
        "optimal_threshold":  optimal_threshold,
        "y_proba":            y_proba
    }


def generate_shap_plot(model, X_test, save_path: str):
    """
    Generate and save a SHAP summary plot for feature importance explainability.
    SHAP (SHapley Additive exPlanations) shows each feature's contribution
    to individual predictions — critical for clinical trust and interviews.

    Args:
        model:     Trained XGBoost classifier.
        X_test:    Test feature matrix (used as background for SHAP).
        save_path: File path to save the PNG plot.
    """
    logger.info("Generating SHAP feature importance plot...")

    explainer   = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)

    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X_test, show=False, max_display=15)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.close()

    logger.info(f"SHAP plot saved to {save_path}")


def train_model():
    """
    Full end-to-end training pipeline:
        1. Load processed feature data
        2. Preprocess and encode
        3. Train/test split (stratified 80/20)
        4. Hyperparameter tuning via GridSearchCV (AUC-optimized)
        5. Evaluate at default and optimal thresholds
        6. Generate SHAP explainability plot
        7. Persist model and feature list to disk
    """
    logger.info("Starting ClinicalIQ Model Training Pipeline...")
    start_time = time.time()

    # 1. Load
    df = load_data(PROCESSED_DATA_PATH)

    # 2. Preprocess
    X, y = preprocess(df)

    # 3. Split
    logger.info("Splitting data — 80% train / 20% test (stratified)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )
    logger.info(f"Train size: {X_train.shape[0]} | Test size: {X_test.shape[0]}")

    # 4. Tune and train
    best_model, best_params, best_cv_auc = tune_and_train(X_train, y_train)

    # 5. Evaluate
    results = evaluate(best_model, X_test, y_test)

    # 6. SHAP plot
    generate_shap_plot(best_model, X_test, SHAP_PLOT_PATH)

    # 7. Save model and feature names
    logger.info(f"Saving model to {MODEL_SAVE_PATH}...")
    joblib.dump(best_model,      MODEL_SAVE_PATH)
    joblib.dump(list(X.columns), FEATURES_SAVE_PATH)
    logger.info("Model and feature list saved successfully.")

    elapsed = round(time.time() - start_time, 2)
    logger.info(f"Pipeline complete in {elapsed} seconds.")
    logger.info(f"Final AUC-ROC : {results['auc']:.4f}")
    logger.info(f"Optimal Threshold : {results['optimal_threshold']:.4f}")


if __name__ == "__main__":
    train_model()