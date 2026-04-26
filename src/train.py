"""
train.py — Train Logistic Regression, Random Forest, and XGBoost on the fraud dataset.
           Prints a side-by-side metrics table and saves the best model (XGBoost) to disk.

Usage:
    python src/train.py
"""

import os
import sys
import pickle
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
)
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

# Allow `python src/train.py` from the project root
sys.path.insert(0, os.path.dirname(__file__))
import preprocess

MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
MODEL_PATH = os.path.join(MODELS_DIR, "xgb_fraud_model.pkl")


# ---------------------------------------------------------------------------
# Model definitions
# ---------------------------------------------------------------------------

def build_models() -> dict:
    """
    All three models use class_weight='balanced' (or the XGBoost equivalent
    scale_pos_weight) as a secondary guard alongside SMOTE.
    scale_pos_weight = legitimate / fraud ≈ 577 for this dataset.
    """
    return {
        "Logistic Regression": LogisticRegression(
            class_weight="balanced",
            max_iter=1000,
            solver="lbfgs",
            random_state=42,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            class_weight="balanced",
            max_depth=12,
            n_jobs=-1,
            random_state=42,
        ),
        "XGBoost": XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            # class_weight equivalent for XGBoost binary classification
            scale_pos_weight=577,
            eval_metric="aucpr",
            use_label_encoder=False,
            random_state=42,
            n_jobs=-1,
        ),
    }


# ---------------------------------------------------------------------------
# Train & evaluate
# ---------------------------------------------------------------------------

def evaluate(model, X_test: np.ndarray, y_test: np.ndarray) -> dict:
    y_pred = model.predict(X_test)
    y_prob = (
        model.predict_proba(X_test)[:, 1]
        if hasattr(model, "predict_proba")
        else model.decision_function(X_test)
    )
    return {
        "Precision": precision_score(y_test, y_pred, zero_division=0),
        "Recall":    recall_score(y_test, y_pred),
        "F1":        f1_score(y_test, y_pred),
        "AUC-ROC":   roc_auc_score(y_test, y_prob),
        "AUC-PR":    average_precision_score(y_test, y_prob),
    }


def train_all(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
) -> tuple[dict, dict]:
    models = build_models()
    results = {}
    trained = {}

    for name, model in models.items():
        print(f"\n  Training {name}...", end=" ", flush=True)
        t0 = time.time()
        model.fit(X_train, y_train)
        elapsed = time.time() - t0
        metrics = evaluate(model, X_test, y_test)
        results[name] = metrics
        trained[name] = model
        print(f"done in {elapsed:.1f}s")

    return results, trained


# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------

def print_comparison_table(results: dict) -> None:
    df = pd.DataFrame(results).T
    df = df[["Precision", "Recall", "F1", "AUC-ROC", "AUC-PR"]]

    sep = "+" + "-" * 24 + "+" + ("-" * 12 + "+") * 5
    header = f"{'Model':<24}{'Precision':>12}{'Recall':>12}{'F1':>12}{'AUC-ROC':>12}{'AUC-PR':>12}"

    print("\n" + "=" * 78)
    print("  MODEL COMPARISON — Test Set (original class distribution, no SMOTE)")
    print("=" * 78)
    print(sep)
    print("| " + header + " |")
    print(sep)
    for model_name, row in df.iterrows():
        line = f"{model_name:<24}"
        for val in row:
            line += f"{val:>12.4f}"
        print("| " + line + " |")
    print(sep)

    best = df["AUC-ROC"].idxmax()
    print(f"\n  Best model by AUC-ROC: {best} ({df.loc[best, 'AUC-ROC']:.4f})")
    print(f"  Best model by Recall : {df['Recall'].idxmax()} ({df['Recall'].max():.4f})")
    print(f"  Best model by AUC-PR : {df['AUC-PR'].idxmax()} ({df['AUC-PR'].max():.4f})")
    print()


# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------

def save_model(model, path: str = MODEL_PATH) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(model, f)
    size_kb = os.path.getsize(path) / 1024
    print(f"  XGBoost model saved → {path}  ({size_kb:.1f} KB)")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    print("\n" + "=" * 78)
    print("  STEP 1 — Preprocessing")
    print("=" * 78)
    X_train, X_test, y_train, y_test, feature_names, _ = preprocess.run()
    print(f"\n  Train shape: {X_train.shape} | Test shape: {X_test.shape}")
    print(f"  Features ({len(feature_names)}): {feature_names[:5]} ... {feature_names[-2:]}")

    print("\n" + "=" * 78)
    print("  STEP 2 — Training")
    print("=" * 78)
    results, trained_models = train_all(X_train, X_test, y_train, y_test)

    print("\n" + "=" * 78)
    print("  STEP 3 — Results")
    print("=" * 78)
    print_comparison_table(results)

    print("=" * 78)
    print("  STEP 4 — Saving XGBoost model")
    print("=" * 78)
    save_model(trained_models["XGBoost"])

    print("\n  Done. Next step: python src/evaluate.py\n")


if __name__ == "__main__":
    main()
