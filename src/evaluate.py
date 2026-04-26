"""
evaluate.py — Evaluate XGBoost fraud model: confusion matrix, ROC curve,
              threshold sweep, SHAP analysis, and fraud_results.csv export.

Usage:
    python src/evaluate.py
"""

import os
import sys
import pickle
import warnings

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix,
    roc_curve,
    auc,
    precision_score,
    recall_score,
    f1_score,
)
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import shap
import matplotlib
matplotlib.use("Agg")   # headless — no display needed
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(__file__))
import preprocess

MODELS_DIR    = os.path.join(os.path.dirname(__file__), "..", "models")
DASHBOARDS_DIR = os.path.join(os.path.dirname(__file__), "..", "dashboards")
MODEL_PATH    = os.path.join(MODELS_DIR, "xgb_fraud_model.pkl")
RESULTS_CSV   = os.path.join(DASHBOARDS_DIR, "fraud_results.csv")

os.makedirs(DASHBOARDS_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def save_plotly(fig: go.Figure, filename: str) -> None:
    path = os.path.join(DASHBOARDS_DIR, filename)
    try:
        fig.write_image(path, scale=2)
        print(f"  Saved → {path}")
    except Exception:
        html_path = path.replace(".png", ".html")
        fig.write_html(html_path)
        print(f"  kaleido unavailable — saved HTML → {html_path}")


# ---------------------------------------------------------------------------
# Step 1 — Load model + reconstruct test set (same split as train.py)
# ---------------------------------------------------------------------------

def load_model(path: str = MODEL_PATH):
    with open(path, "rb") as f:
        model = pickle.load(f)
    print(f"  Model loaded from {path}")
    return model


def build_test_set() -> tuple:
    """
    Replicate the exact train/test split from preprocess.py so evaluate.py
    can map test rows back to original (unscaled) Amount and Hour values.
    """
    df_raw   = preprocess.load_data()
    df_feat  = preprocess.engineer_features(df_raw)       # has original Amount, Hour
    df_scaled, _ = preprocess.scale_features(df_feat)    # scales Amount, Hour in-place copy

    feature_names = df_scaled.drop(columns=["Class"]).columns.tolist()
    X = df_scaled.drop(columns=["Class"]).values
    y = df_scaled["Class"].values

    # Replicate the same stratified split — identical to split_and_resample()
    _, X_test, _, y_test, _, idx_test = train_test_split(
        X, y, df_feat.index,
        test_size=0.2, stratify=y, random_state=42
    )

    # Original (pre-scale) amounts and hours for the CSV export
    original_amount = df_feat.loc[idx_test, "Amount"].values
    original_hour   = df_feat.loc[idx_test, "Hour"].values

    return X_test, y_test, idx_test.values, original_amount, original_hour, feature_names


# ---------------------------------------------------------------------------
# Step 2 — Confusion matrix
# ---------------------------------------------------------------------------

def plot_confusion_matrix(y_test: np.ndarray, y_pred: np.ndarray) -> None:
    cm = confusion_matrix(y_test, y_pred)
    labels = ["Legitimate", "Fraud"]

    # Annotation text: count + percentage of actual class
    text = [[
        f"{cm[r][c]:,}<br>({cm[r][c] / cm[r].sum():.1%})"
        for c in range(2)
    ] for r in range(2)]

    fig = go.Figure(go.Heatmap(
        z=cm,
        x=[f"Predicted {l}" for l in labels],
        y=[f"Actual {l}" for l in labels],
        text=text,
        texttemplate="%{text}",
        colorscale="Blues",
        showscale=True,
        colorbar=dict(title="Count"),
    ))
    fig.update_layout(
        title="Confusion Matrix — XGBoost (threshold = 0.5)",
        xaxis_title="Predicted Label",
        yaxis_title="Actual Label",
        template="plotly_white",
        width=600, height=500,
        font=dict(size=14),
    )
    save_plotly(fig, "confusion_matrix.png")


# ---------------------------------------------------------------------------
# Step 3 — ROC curve
# ---------------------------------------------------------------------------

def plot_roc_curve(y_test: np.ndarray, y_prob: np.ndarray) -> None:
    fpr, tpr, thresholds = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)

    # Find point closest to top-left corner (optimal threshold)
    dist = np.sqrt(fpr ** 2 + (1 - tpr) ** 2)
    best_idx = np.argmin(dist)

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=fpr, y=tpr,
        mode="lines",
        name=f"XGBoost (AUC = {roc_auc:.4f})",
        line=dict(color="#DD5B5B", width=2.5),
        fill="tozeroy",
        fillcolor="rgba(221,91,91,0.1)",
    ))
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1],
        mode="lines",
        name="Random Classifier",
        line=dict(color="grey", dash="dash"),
    ))
    fig.add_trace(go.Scatter(
        x=[fpr[best_idx]], y=[tpr[best_idx]],
        mode="markers+text",
        name=f"Optimal threshold ({thresholds[best_idx]:.2f})",
        marker=dict(color="#4C72B0", size=10, symbol="star"),
        text=[f"  thresh={thresholds[best_idx]:.2f}"],
        textposition="middle right",
    ))
    fig.update_layout(
        title=f"ROC Curve — XGBoost  (AUC = {roc_auc:.4f})",
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate (Recall)",
        template="plotly_white",
        width=700, height=550,
        legend=dict(x=0.55, y=0.15),
        font=dict(size=13),
    )
    save_plotly(fig, "roc_curve.png")
    return roc_auc


# ---------------------------------------------------------------------------
# Step 4 — Threshold sweep
# ---------------------------------------------------------------------------

def threshold_sweep(y_test: np.ndarray, y_prob: np.ndarray) -> float:
    thresholds = np.arange(0.1, 1.0, 0.1)
    rows = []

    for t in thresholds:
        y_pred_t = (y_prob >= t).astype(int)
        rows.append({
            "Threshold": round(t, 1),
            "Precision": precision_score(y_test, y_pred_t, zero_division=0),
            "Recall":    recall_score(y_test, y_pred_t),
            "F1":        f1_score(y_test, y_pred_t, zero_division=0),
            "Fraud_Flagged": y_pred_t.sum(),
        })

    df = pd.DataFrame(rows)
    best_row = df.loc[df["F1"].idxmax()]

    sep = "+" + "-" * 12 + "+" + ("-" * 12 + "+") * 4
    header = f"{'Threshold':>12}{'Precision':>12}{'Recall':>12}{'F1':>12}{'Flagged':>12}"
    print("\n" + "=" * 62)
    print("  THRESHOLD SWEEP (XGBoost, test set)")
    print("=" * 62)
    print(sep)
    print("| " + header + " |")
    print(sep)
    for _, row in df.iterrows():
        marker = " ◄ best F1" if row["Threshold"] == best_row["Threshold"] else ""
        line = (
            f"{row['Threshold']:>12.1f}"
            f"{row['Precision']:>12.4f}"
            f"{row['Recall']:>12.4f}"
            f"{row['F1']:>12.4f}"
            f"{int(row['Fraud_Flagged']):>12,}"
        )
        print("| " + line + " |" + marker)
    print(sep)
    print(f"\n  Optimal threshold by F1: {best_row['Threshold']:.1f}")
    print(f"    Precision={best_row['Precision']:.4f}  "
          f"Recall={best_row['Recall']:.4f}  "
          f"F1={best_row['F1']:.4f}\n")

    return float(best_row["Threshold"])


# ---------------------------------------------------------------------------
# Step 5 — SHAP
# ---------------------------------------------------------------------------

def run_shap(
    model,
    X_test: np.ndarray,
    feature_names: list[str],
) -> np.ndarray:
    import xgboost as xgb
    print("  Computing SHAP values (XGBoost native pred_contribs)...")
    # Use XGBoost's built-in SHAP to avoid shap-library/XGBoost 3.x version skew.
    # pred_contribs returns (n_samples, n_features + 1) — last col is the bias term.
    dmatrix     = xgb.DMatrix(X_test, feature_names=feature_names)
    raw         = model.get_booster().predict(dmatrix, pred_contribs=True)
    shap_values = raw[:, :-1]   # drop bias

    # Summary bar plot — top 15 features
    plt.figure(figsize=(10, 7))
    shap.summary_plot(
        shap_values, X_test,
        feature_names=feature_names,
        plot_type="bar",
        max_display=15,
        show=False,
    )
    plt.title("SHAP Feature Importance — XGBoost", fontsize=14, pad=12)
    plt.tight_layout()
    bar_path = os.path.join(DASHBOARDS_DIR, "shap_summary_bar.png")
    plt.savefig(bar_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {bar_path}")

    # Beeswarm summary plot
    plt.figure(figsize=(10, 7))
    shap.summary_plot(
        shap_values, X_test,
        feature_names=feature_names,
        max_display=15,
        show=False,
    )
    plt.title("SHAP Beeswarm — XGBoost", fontsize=14, pad=12)
    plt.tight_layout()
    bee_path = os.path.join(DASHBOARDS_DIR, "shap_summary_beeswarm.png")
    plt.savefig(bee_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {bee_path}")

    return shap_values


# ---------------------------------------------------------------------------
# Step 6 — Export fraud_results.csv
# ---------------------------------------------------------------------------

def export_results(
    idx_test: np.ndarray,
    original_amount: np.ndarray,
    original_hour: np.ndarray,
    y_prob: np.ndarray,
    shap_values: np.ndarray,
    feature_names: list[str],
    optimal_threshold: float,
) -> None:
    # Top SHAP feature per transaction (highest |shap value|)
    top_idx        = np.argmax(np.abs(shap_values), axis=1)
    top_features   = np.array(feature_names)[top_idx]
    top_shap_vals  = shap_values[np.arange(len(shap_values)), top_idx]

    df_out = pd.DataFrame({
        "transaction_id":    idx_test,
        "amount":            np.round(original_amount, 4),
        "hour":              original_hour.astype(int),
        "predicted_label":   (y_prob >= optimal_threshold).astype(int),
        "fraud_probability": np.round(y_prob, 6),
        "top_shap_feature":  top_features,
        "shap_value":        np.round(top_shap_vals, 6),
    })

    df_out.to_csv(RESULTS_CSV, index=False)
    fraud_flagged = df_out["predicted_label"].sum()
    print(f"\n  fraud_results.csv saved → {RESULTS_CSV}")
    print(f"  Rows: {len(df_out):,}  |  Fraud flagged: {fraud_flagged:,} "
          f"({fraud_flagged / len(df_out):.4%})  |  Threshold: {optimal_threshold}")
    print(f"\n  Preview:")
    print(df_out.head(5).to_string(index=False))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    print("\n" + "=" * 62)
    print("  STEP 1 — Load model + reconstruct test set")
    print("=" * 62)
    model = load_model()
    X_test, y_test, idx_test, orig_amount, orig_hour, feature_names = build_test_set()
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)
    print(f"  Test set: {X_test.shape[0]:,} rows | "
          f"True fraud: {y_test.sum():,} | "
          f"Features: {len(feature_names)}")

    print("\n" + "=" * 62)
    print("  STEP 2 — Confusion matrix")
    print("=" * 62)
    plot_confusion_matrix(y_test, y_pred)

    print("\n" + "=" * 62)
    print("  STEP 3 — ROC curve")
    print("=" * 62)
    roc_auc = plot_roc_curve(y_test, y_prob)
    print(f"  AUC-ROC: {roc_auc:.4f}")

    print("\n" + "=" * 62)
    print("  STEP 4 — Threshold sweep")
    print("=" * 62)
    optimal_threshold = threshold_sweep(y_test, y_prob)

    print("=" * 62)
    print("  STEP 5 — SHAP analysis")
    print("=" * 62)
    shap_values = run_shap(model, X_test, feature_names)

    print("\n" + "=" * 62)
    print("  STEP 6 — Export fraud_results.csv")
    print("=" * 62)
    export_results(
        idx_test, orig_amount, orig_hour,
        y_prob, shap_values, feature_names,
        optimal_threshold,
    )

    print("\n" + "=" * 62)
    print("  Done. Outputs in dashboards/:")
    for f in sorted(os.listdir(DASHBOARDS_DIR)):
        size_kb = os.path.getsize(os.path.join(DASHBOARDS_DIR, f)) / 1024
        print(f"    {f:<35} {size_kb:>8.1f} KB")
    print()
    print("  Next step: app/streamlit_app.py\n")


if __name__ == "__main__":
    main()
