"""
streamlit_app.py — Real-time credit card fraud scoring.
Upload a CSV (creditcard.csv format) or use the built-in sample.

Run:  streamlit run app/streamlit_app.py
"""

import os
import sys
import pickle
import warnings

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import xgboost as xgb

warnings.filterwarnings("ignore")

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT        = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH  = os.path.join(ROOT, "models", "xgb_fraud_model.pkl")
SCALER_PATH = os.path.join(ROOT, "models", "scaler.pkl")
DATA_PATH   = os.path.join(ROOT, "data", "creditcard.csv")

# Must match column order used in train.py exactly
FEATURE_COLS = [f"V{i}" for i in range(1, 29)] + ["Amount", "Hour"]

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Fraud Detection",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  [data-testid="metric-container"] {
    background: #f8f9fa;
    border: 1px solid #e0e0e0;
    border-radius: 8px;
    padding: 16px 20px;
  }
  .fraud-badge  { color: #c0392b; font-weight: 600; }
  .safe-badge   { color: #1e8449; font-weight: 600; }
  div[data-testid="stDownloadButton"] > button {
    width: 100%;
    margin-top: 8px;
  }
</style>
""", unsafe_allow_html=True)


# ── Cached loaders ─────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading model…")
def load_artifacts():
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    with open(SCALER_PATH, "rb") as f:
        scaler = pickle.load(f)
    return model, scaler


@st.cache_data(show_spinner="Loading sample data…")
def load_sample(n_legit: int = 470, n_fraud: int = 30) -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    legit = df[df["Class"] == 0].sample(n_legit, random_state=42)
    fraud = df[df["Class"] == 1].sample(n_fraud, random_state=42)
    return (
        pd.concat([legit, fraud])
        .sample(frac=1, random_state=42)
        .reset_index(drop=True)
    )


# ── Preprocessing ──────────────────────────────────────────────────────────────
def prepare(df_raw: pd.DataFrame, scaler) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """
    Returns (df_scaled_features, orig_amount, orig_hour).
    Accepts both creditcard.csv-format (with Time) and pre-processed files (with Hour).
    """
    df = df_raw.copy()

    if "Time" in df.columns:
        df["Hour"] = (df["Time"] // 3600) % 24
        df.drop(columns=["Time"], inplace=True)
    elif "Hour" not in df.columns:
        raise ValueError("Input must contain either a 'Time' or 'Hour' column.")

    missing = [c for c in FEATURE_COLS if c not in df.columns]
    if missing:
        needed = ", ".join(f"`{c}`" for c in missing)
        raise ValueError(
            f"**Your file is missing {len(missing)} required column(s):** {needed}\n\n"
            f"The app expects: `Time` (or `Hour`), `V1`–`V28`, `Amount`. "
            f"`Class` is optional. Make sure you're uploading a standard "
            f"`creditcard.csv`-format file from Kaggle."
        )

    orig_amount = df_raw["Amount"].values.copy()
    orig_hour   = df["Hour"].values.copy().astype(int)

    df[["Amount", "Hour"]] = scaler.transform(df[["Amount", "Hour"]])
    return df, orig_amount, orig_hour


# ── Inference ──────────────────────────────────────────────────────────────────
def run_model(model, df_scaled: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    X = df_scaled[FEATURE_COLS].values
    y_prob = model.predict_proba(X)[:, 1]

    dmatrix  = xgb.DMatrix(X, feature_names=FEATURE_COLS)
    raw_shap = model.get_booster().predict(dmatrix, pred_contribs=True)
    shap_vals = raw_shap[:, :-1]   # drop bias term
    return y_prob, shap_vals


# ── KPI cards ──────────────────────────────────────────────────────────────────
# 0.17% = baseline fraud rate in the full Kaggle dataset
_BASELINE_FRAUD_RATE = 0.17
# $88.35 = approximate mean fraud transaction amount in the full dataset
_BASELINE_FRAUD_AMT  = 88.35

def render_kpis(df_out: pd.DataFrame) -> None:
    total    = len(df_out)
    flagged  = int(df_out["predicted_label"].sum())
    rate     = flagged / total * 100
    fraud_df = df_out[df_out["predicted_label"] == 1]
    avg_amt  = fraud_df["amount"].mean() if len(fraud_df) else 0.0

    rate_delta     = rate - _BASELINE_FRAUD_RATE          # positive → above baseline → red
    amt_delta      = avg_amt - _BASELINE_FRAUD_AMT

    c1, c2, c3, c4 = st.columns(4)
    c1.metric(
        "Total Transactions",
        f"{total:,}",
    )
    c2.metric(
        "Flagged as Fraud",
        f"{flagged:,}",
        delta=f"{rate:.2f}% of batch",
        delta_color="inverse",   # any fraud flagging → red arrow
        help="Transactions whose predicted fraud probability meets or exceeds the threshold.",
    )
    c3.metric(
        "Fraud Rate",
        f"{rate:.2f}%",
        delta=f"{rate_delta:+.2f}pp vs dataset avg ({_BASELINE_FRAUD_RATE:.2f}%)",
        delta_color="inverse",   # above baseline → red; below → green
        help="Percentage of transactions flagged. Dataset average is 0.17%.",
    )
    c4.metric(
        "Avg Fraud Amount",
        f"${avg_amt:,.2f}",
        delta=f"${amt_delta:+.2f} vs dataset avg (${_BASELINE_FRAUD_AMT:.2f})",
        delta_color="inverse",
        help="Mean transaction amount among flagged rows. Dataset fraud average is ~$88.",
    )


# ── SHAP bar chart ─────────────────────────────────────────────────────────────
def render_shap_chart(shap_vals: np.ndarray, df_out: pd.DataFrame) -> None:
    # Compute mean |SHAP| for fraud-flagged transactions only (more diagnostic)
    fraud_mask = df_out["predicted_label"].values == 1
    base = shap_vals[fraud_mask] if fraud_mask.any() else shap_vals
    mean_abs = np.abs(base).mean(axis=0)

    top_n   = min(15, len(FEATURE_COLS))
    top_idx = np.argsort(mean_abs)[::-1][:top_n]
    top_vals = mean_abs[top_idx]
    top_names = [FEATURE_COLS[i] for i in top_idx]

    fig = go.Figure(go.Bar(
        x=top_vals[::-1],
        y=top_names[::-1],
        orientation="h",
        marker=dict(
            color=top_vals[::-1],
            colorscale="Reds",
            showscale=False,
        ),
        text=[f"{v:.3f}" for v in top_vals[::-1]],
        textposition="outside",
    ))
    fig.update_layout(
        title="Top SHAP Features (Fraud-Flagged Rows)",
        xaxis_title="Mean |SHAP Value|",
        template="plotly_white",
        height=420,
        margin=dict(l=10, r=60, t=50, b=40),
        font=dict(size=12),
    )
    st.plotly_chart(fig, use_container_width=True)


# ── Probability histogram ──────────────────────────────────────────────────────
def render_prob_histogram(df_out: pd.DataFrame, threshold: float) -> None:
    # Colour by true label if available, else by predicted label
    label_col  = "true_label" if "true_label" in df_out.columns else "predicted_label"
    label_map  = {0: ("Legitimate", "#4C72B0"), 1: ("Fraud", "#DD5B5B")}

    fig = go.Figure()
    for cls, (name, color) in label_map.items():
        subset = df_out[df_out[label_col] == cls]["fraud_probability"]
        if len(subset):
            fig.add_trace(go.Histogram(
                x=subset,
                name=name,
                marker_color=color,
                opacity=0.72,
                nbinsx=50,
            ))

    fig.add_vline(
        x=threshold,
        line_dash="dash",
        line_color="#2c3e50",
        line_width=2,
        annotation_text=f"  Threshold = {threshold:.2f}",
        annotation_position="top right",
        annotation_font=dict(size=12, color="#2c3e50"),
    )
    color_note = "true" if label_col == "true_label" else "predicted"
    fig.update_layout(
        title=f"Fraud Probability Distribution (coloured by {color_note} label)",
        xaxis_title="Predicted Fraud Probability",
        yaxis_title="Count",
        barmode="overlay",
        template="plotly_white",
        height=420,
        legend=dict(x=0.01, y=0.97),
        margin=dict(l=10, r=20, t=50, b=40),
        font=dict(size=12),
    )
    st.plotly_chart(fig, use_container_width=True)


# ── Transaction table ──────────────────────────────────────────────────────────
def render_table(df_out: pd.DataFrame) -> None:
    st.subheader("Transaction Results")

    display_cols = [
        "transaction_id", "amount", "hour",
        "fraud_probability", "predicted_label", "top_shap_feature",
    ]
    if "true_label" in df_out.columns:
        display_cols.insert(5, "true_label")

    disp = df_out[display_cols].copy()
    disp["amount"] = disp["amount"].round(2)
    disp["fraud_probability"] = disp["fraud_probability"].round(4)

    def _style(row):
        bg = "#ffdddd" if row["predicted_label"] == 1 else "#e8f8f0"
        fg = "#7b241c" if row["predicted_label"] == 1 else "#1d6a3b"
        return [f"background-color:{bg}; color:{fg}"] * len(row)

    styled = (
        disp.style
        .apply(_style, axis=1)
        .format({"amount": "${:.2f}", "fraud_probability": "{:.4f}"})
    )
    st.dataframe(styled, use_container_width=True, height=420)

    csv_bytes = df_out.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="⬇ Download Full Results CSV",
        data=csv_bytes,
        file_name="fraud_results.csv",
        mime="text/csv",
    )


# ── "How does this work?" expander ────────────────────────────────────────────
def render_explainer() -> None:
    with st.expander("How does this work?", icon="💡"):
        st.markdown("""
**This app uses three key techniques working together to flag suspicious credit card transactions.**

---

#### 🔁 SMOTE — Handling the Needle-in-a-Haystack Problem
Real fraud data is wildly imbalanced: in this dataset, only **1 in 578 transactions** is fraud.
If we trained a model naively, it could score 99.8% accuracy just by predicting "not fraud" every
single time — and it would still miss every fraudster.

**SMOTE** (Synthetic Minority Over-sampling TEchnique) fixes this by generating *synthetic*
fraud examples that are similar to real ones, balancing the training set so the model actually
learns what fraud looks like rather than ignoring it.

---

#### 🌲 XGBoost — The Model Making the Predictions
**XGBoost** (eXtreme Gradient Boosting) builds hundreds of small decision trees one after another,
with each tree correcting the mistakes of the one before it. Think of it as 300 fraud investigators
reviewing every transaction in sequence, each one focused on the cases the previous ones got wrong.

The result is a model that outputs a **fraud probability between 0 and 1** for each transaction.
The **Risk Threshold** slider in the sidebar controls where you draw the line — lower threshold
catches more fraud but raises more false alarms; higher threshold does the reverse.

---

#### 🔍 SHAP — Explaining *Why* a Transaction Was Flagged
Even a perfect model is useless in a compliance context if you can't explain its decisions.
**SHAP** (SHapley Additive exPlanations) is a technique from game theory that measures how much
each feature *pushed* the model's prediction toward fraud or away from it — for every single
transaction individually.

The **Top SHAP Features** chart above shows which variables most influenced the fraud-flagged
transactions in your current batch. A feature like `V14` having a high value means that the model
weighted it heavily when deciding to flag those rows.

---

*Model: XGBoost trained on the [Kaggle Credit Card Fraud dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
(284,807 transactions, 492 fraud cases). Metrics: AUC-ROC 0.979 · AUC-PR 0.840.*
        """)


# ── Sidebar ────────────────────────────────────────────────────────────────────
def build_sidebar() -> tuple[float, pd.DataFrame | None]:
    with st.sidebar:
        st.image(
            "https://img.icons8.com/color/96/null/error--v1.png",
            width=60,
        )
        st.title("Fraud Detection")
        st.caption("XGBoost · SHAP · Streamlit")
        st.divider()

        st.subheader("Risk Threshold")
        threshold = st.slider(
            "Flag transactions above this probability as fraud",
            min_value=0.0, max_value=1.0,
            value=0.5, step=0.01,
            help="Lower = catch more fraud (higher recall, lower precision).\n"
                 "Higher = fewer false alarms (higher precision, lower recall).",
        )
        # Live precision / recall hint
        st.caption(
            "↓ threshold → more fraud caught, more false alarms  \n"
            "↑ threshold → fewer false alarms, more misses"
        )
        st.divider()

        st.subheader("Data Source")
        source = st.radio(
            "Input",
            ["Built-in sample (500 rows)", "Upload CSV"],
            label_visibility="collapsed",
        )

        uploaded = None
        if source == "Upload CSV":
            uploaded = st.file_uploader(
                "creditcard.csv-format file",
                type=["csv"],
                help="Must contain columns: Time (or Hour), V1–V28, Amount. "
                     "Class column is optional.",
            )

        st.divider()
        st.caption(
            "**Required columns:**  \n"
            "`Time` or `Hour`, `V1`–`V28`, `Amount`  \n"
            "`Class` is optional (used for chart colouring)"
        )

    return threshold, uploaded


# ── Main ───────────────────────────────────────────────────────────────────────
def main() -> None:
    threshold, uploaded = build_sidebar()

    st.title("🔍 Credit Card Fraud Detection")
    st.info(
        "Upload a batch of credit card transactions and this app will instantly score each one "
        "for fraud risk using a trained XGBoost model. "
        "Use the **Risk Threshold** slider in the sidebar to tune how aggressively fraud is flagged — "
        "lower values catch more fraud but generate more false alarms.",
        icon="ℹ️",
    )

    # Load model + scaler
    try:
        model, scaler = load_artifacts()
    except FileNotFoundError as exc:
        st.error(f"**Model not found:** {exc}  \nRun `python src/train.py` first.")
        st.stop()

    # Load data
    if uploaded is not None:
        with st.spinner("Reading uploaded file…"):
            df_raw = pd.read_csv(uploaded)
        st.success(f"Uploaded file loaded — {len(df_raw):,} transactions.")
    else:
        if not os.path.exists(DATA_PATH):
            st.error(
                "Sample data not found at `data/creditcard.csv`.  \n"
                "Download from Kaggle or switch to **Upload CSV**."
            )
            st.stop()
        df_raw = load_sample()
        st.info(
            "Using built-in sample: **470 legitimate + 30 fraud** transactions "
            "drawn randomly from `data/creditcard.csv`."
        )

    # Preprocess
    try:
        df_scaled, orig_amount, orig_hour = prepare(df_raw, scaler)
    except ValueError as exc:
        st.error(str(exc))
        st.stop()

    # Run model
    with st.spinner("Running predictions and SHAP analysis…"):
        y_prob, shap_vals = run_model(model, df_scaled)

    # Build output df
    top_shap_idx  = np.argmax(np.abs(shap_vals), axis=1)
    top_shap_feat = np.array(FEATURE_COLS)[top_shap_idx]

    df_out = pd.DataFrame({
        "transaction_id":    df_raw.index,
        "amount":            orig_amount,
        "hour":              orig_hour,
        "predicted_label":   (y_prob >= threshold).astype(int),
        "fraud_probability": np.round(y_prob, 6),
        "top_shap_feature":  top_shap_feat,
    })
    if "Class" in df_raw.columns:
        df_out["true_label"] = df_raw["Class"].values

    # ── Layout ────────────────────────────────────────────────────────────────
    st.divider()

    # KPI row
    render_kpis(df_out)

    st.divider()

    # Charts row
    col_left, col_right = st.columns(2)
    with col_left:
        render_shap_chart(shap_vals, df_out)
    with col_right:
        render_prob_histogram(df_out, threshold)

    st.divider()

    # Table + download
    render_table(df_out)

    st.divider()

    # Plain-English explainer for non-technical viewers
    render_explainer()


if __name__ == "__main__":
    main()
