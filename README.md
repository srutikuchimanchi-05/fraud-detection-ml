# 🔍 Credit Card Fraud Detection

![Python](https://img.shields.io/badge/Python-3.13-3776AB?style=flat&logo=python&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-3.2-FF6600?style=flat&logo=xgboost&logoColor=white)
![SHAP](https://img.shields.io/badge/SHAP-0.49-FF4B4B?style=flat)
![Streamlit](https://img.shields.io/badge/Streamlit-1.35-FF4B4B?style=flat&logo=streamlit&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.5-F7931E?style=flat&logo=scikit-learn&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green?style=flat)

An end-to-end machine learning pipeline that detects fraudulent credit card transactions in real time. Trained on 284,807 transactions with a 0.17% fraud rate, the system uses SMOTE oversampling, XGBoost classification, and SHAP explainability — deployed as an interactive Streamlit app with Tableau and Power BI dashboards.

---

## The Business Problem

> **Credit card fraud costs the global economy an estimated $32 billion per year** — and that number is growing as digital payments expand. Traditional rule-based fraud systems generate high false-positive rates, frustrating legitimate customers while still missing sophisticated fraud patterns. Machine learning offers a better path: models that learn the statistical signature of fraud from historical data and score every new transaction in milliseconds.

The challenge is that fraud data is **severely imbalanced** — legitimate transactions outnumber fraud nearly 578-to-1 in this dataset. A naive classifier that predicts "not fraud" on every transaction achieves 99.83% accuracy while catching exactly zero fraudsters. This project addresses that directly using SMOTE oversampling and class-weight balancing, optimizing for Recall and AUC-PR rather than raw accuracy.

---

## Live Demos

| Platform | Link |
|---|---|
| 🚀 **Streamlit App** | *Coming soon — deploying to Streamlit Community Cloud* |
| 📊 **Tableau Public Dashboard** | *Coming soon — Fraud Intelligence Center* |
| 📈 **Power BI Report** | *Coming soon — Fraud Executive Dashboard* |

---

## Model Results

All models evaluated on a held-out test set of 56,962 transactions preserving the original class distribution (no SMOTE on test set). Primary metrics are **AUC-PR** and **Recall** — accuracy is not reported as it is misleading on imbalanced data.

| Model | Precision | Recall | F1 | AUC-ROC | AUC-PR |
|---|---|---|---|---|---|
| Logistic Regression | 0.06 | 0.92 | 0.11 | 0.97 | 0.73 |
| Random Forest | 0.53 | 0.85 | 0.65 | 0.98 | 0.83 |
| **XGBoost** ✓ | **0.13** | **0.90** | **0.24** | **0.98** | **0.84** |

XGBoost was selected as the production model for its best AUC-PR (0.84) and strong recall (0.90), indicating it correctly identifies 9 out of 10 fraud cases. Random Forest achieves better F1 at the default 0.5 threshold but XGBoost's probability calibration gives more control via threshold tuning.

### Threshold Optimization

Because fraud has asymmetric costs (a missed fraud is more damaging than a false alarm), the decision threshold can be tuned independently of the model. XGBoost at threshold **0.9** gives the best F1:

| Threshold | Precision | Recall | F1 | Flagged |
|---|---|---|---|---|
| 0.1 | 0.05 | 0.91 | 0.10 | 1,717 |
| 0.3 | 0.09 | 0.90 | 0.17 | 944 |
| 0.5 | 0.14 | 0.90 | 0.24 | 647 |
| 0.7 | 0.19 | 0.89 | 0.32 | 449 |
| **0.9** | **0.33** | **0.87** | **0.48** | **259** |

The Streamlit app exposes this as a live slider — users can set the threshold based on their operational risk tolerance.

---

## Architecture

```
Transaction batch (CSV)
        │
        ▼
┌───────────────────┐
│   preprocess.py   │  StandardScaler (Amount, Hour)
│                   │  SMOTE (training set only)
│                   │  Stratified 80/20 split
└────────┬──────────┘
         │
         ▼
┌───────────────────┐
│    train.py       │  Logistic Regression
│                   │  Random Forest
│                   │  XGBoost ← saved to models/
└────────┬──────────┘
         │
         ▼
┌───────────────────┐
│   evaluate.py     │  Confusion matrix + ROC curve (Plotly → PNG)
│                   │  Threshold sweep (0.1 → 0.9)
│                   │  SHAP values (XGBoost native pred_contribs)
│                   │  fraud_results.csv → dashboards/
└────────┬──────────┘
         │
    ┌────┴──────────────────────────┐
    ▼                               ▼
┌──────────────┐           ┌─────────────────┐
│ Streamlit    │           │  Tableau Public  │
│ app.py       │           │  + Power BI      │
│ (live UI)    │           │  (fraud_results  │
└──────────────┘           │   .csv as source)│
                           └─────────────────┘
```

---

## Folder Structure

```
fraud-detection/
├── data/
│   └── creditcard.csv              ← download from Kaggle (not committed)
├── notebooks/
│   ├── 01_eda.ipynb                ← class imbalance, correlations, time/amount patterns
│   └── 02_modeling.ipynb           ← model training, evaluation, SHAP
├── src/
│   ├── preprocess.py               ← StandardScaler, SMOTE, train/test split
│   ├── train.py                    ← train LR, RF, XGBoost; print comparison table
│   └── evaluate.py                 ← confusion matrix, ROC, threshold sweep, SHAP, CSV export
├── dashboards/
│   ├── fraud_results.csv           ← exported predictions + SHAP for BI tools
│   ├── confusion_matrix.png
│   ├── roc_curve.png
│   ├── shap_summary_bar.png
│   └── shap_summary_beeswarm.png
├── models/
│   ├── xgb_fraud_model.pkl         ← trained XGBoost model
│   └── scaler.pkl                  ← fitted StandardScaler
├── tests/
│   └── test_preprocess.py
├── app/
│   └── streamlit_app.py            ← upload CSV → predictions + SHAP explanations
├── requirements.txt
├── README.md
└── .gitignore
```

---

## Quick Start

### 1. Clone the repo

```bash
git clone https://github.com/YOUR_USERNAME/fraud-detection.git
cd fraud-detection
```

### 2. Download the dataset

Download `creditcard.csv` from [Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) and place it at:

```
data/creditcard.csv
```

### 3. Create a virtual environment

```bash
# macOS / Linux
python3 -m venv .venv
source .venv/bin/activate

# Windows
python -m venv .venv
.venv\Scripts\activate
```

> **macOS only:** XGBoost requires OpenMP. Install it once with:
> ```bash
> brew install libomp
> ```

### 4. Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 5. Run the pipeline

```bash
# Step 1 — preprocess (scales data, applies SMOTE, saves scaler)
python src/preprocess.py

# Step 2 — train all three models and save XGBoost
python src/train.py

# Step 3 — evaluate, generate plots, export fraud_results.csv
python src/evaluate.py
```

### 6. Launch the Streamlit app

```bash
streamlit run app/streamlit_app.py
```

Open **http://localhost:8501** in your browser. Use the built-in sample or upload your own `creditcard.csv`-format file.

---

## Key Technical Decisions

| Decision | Rationale |
|---|---|
| **SMOTE on training set only** | Applying SMOTE after splitting prevents data leakage — the test set reflects real-world class distribution |
| **AUC-PR as primary metric** | More informative than AUC-ROC on severely imbalanced data; measures precision across all recall thresholds |
| **`scale_pos_weight=577` in XGBoost** | Equivalent to `class_weight='balanced'` — weights the minority class by the legitimate/fraud ratio |
| **XGBoost native SHAP** (`pred_contribs=True`) | Avoids the shap-library/XGBoost 3.x version incompatibility; faster for large batches |
| **Threshold at 0.9 for best F1** | Default 0.5 threshold under-penalizes false positives; 0.9 improves precision without sacrificing meaningful recall |
| **StandardScaler on Amount + Hour only** | V1–V28 are already PCA-transformed and unit-scaled by the dataset provider |

---

## Dashboard Plans

### Tableau Public — "Fraud Intelligence Center"
- Precision/Recall curve with live threshold parameter slider
- Hourly fraud rate heatmap (LOD expression)
- SHAP feature importance bar chart with set actions
- Cumulative fraud caught / lift curve (table calculation)
- Fraud amount distribution (dual-axis histogram + boxplot)

### Power BI — "Fraud Executive Dashboard"
- What-If parameter: risk threshold slider → live Precision/Recall DAX
- Key Influencers visual (AI-driven, no-code)
- Decomposition Tree: fraud → hour → amount range → top SHAP feature
- Smart Narrative auto-summary
- Drillthrough page: per-transaction SHAP breakdown

Both dashboards use `dashboards/fraud_results.csv` as their data source, with columns:
`transaction_id · amount · hour · predicted_label · fraud_probability · top_shap_feature · shap_value`

---

## Dataset

**[Kaggle Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)**
- 284,807 transactions over two days in September 2013 (European cardholders)
- 492 fraud cases — **0.17% of all transactions**
- Features V1–V28 are PCA components (anonymized for confidentiality)
- Only `Time`, `Amount`, and `Class` are in their original form

---

## Tech Stack

| Layer | Tools |
|---|---|
| Data & ML | Python · pandas · NumPy · scikit-learn · imbalanced-learn |
| Modeling | XGBoost · Logistic Regression · Random Forest |
| Explainability | SHAP |
| Visualization | Plotly · Seaborn · Matplotlib |
| App | Streamlit |
| BI | Tableau Public · Power BI Desktop |
| Deployment | Streamlit Community Cloud |

---

## Author

**Sruti Kuchimanchi**
[GitHub](https://github.com/YOUR_USERNAME) · [LinkedIn](https://linkedin.com/in/YOUR_HANDLE)

---

*Built as a portfolio project demonstrating end-to-end ML engineering: data imbalance handling, model selection, explainability, and interactive deployment.*
