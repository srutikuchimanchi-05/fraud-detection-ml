# Project: Credit Card Fraud Detection System
**Owner:** Sruti Kuchimanchi
**Goal:** Build an end-to-end fraud detection ML pipeline with advanced Tableau Public and Power BI dashboards. Deploy as a Streamlit app. Publish to GitHub.

---

## Stack (All Free)
- **Language:** Python 3.x
- **Data:** Kaggle Credit Card Fraud Detection dataset (`data/creditcard.csv`)
- **ML:** scikit-learn, XGBoost, imbalanced-learn (SMOTE)
- **Explainability:** SHAP
- **Visualization (code):** Plotly, Seaborn, Matplotlib
- **BI:** Tableau Public + Power BI Desktop
- **App:** Streamlit
- **Deployment:** Streamlit Community Cloud (free)
- **Tracking:** Git + GitHub

---

## Folder Structure
```
fraud-detection/
├── data/
│   └── creditcard.csv              ← download from Kaggle first
├── notebooks/
│   ├── 01_eda.ipynb                ← EDA: class imbalance, correlations, time/amount patterns
│   └── 02_modeling.ipynb           ← Model training, evaluation, SHAP
├── src/
│   ├── preprocess.py               ← StandardScaler, SMOTE, train/test split
│   ├── train.py                    ← Train LR, RF, XGBoost; log metrics
│   └── evaluate.py                 ← Precision, Recall, F1, AUC-ROC, confusion matrix
├── dashboards/
│   └── fraud_results.csv           ← exported from Python for Tableau + Power BI
├── models/
│   └── xgb_fraud_model.pkl         ← saved trained model
├── tests/
│   └── test_preprocess.py
├── app/
│   └── streamlit_app.py            ← upload CSV → get predictions + SHAP explanations
├── requirements.txt
├── README.md
└── .gitignore
```

---

## Build Order (follow this sequence)
1. `src/preprocess.py` — load data, scale, SMOTE, split
2. `notebooks/01_eda.ipynb` — EDA with Plotly/Seaborn
3. `src/train.py` — train 3 models, compare metrics
4. `src/evaluate.py` — evaluation + SHAP export
5. `dashboards/fraud_results.csv` — export for BI tools
6. `app/streamlit_app.py` — Streamlit scoring app
7. `README.md` — with Tableau + Power BI links

---

## Key Technical Decisions
- **Imbalance handling:** Use SMOTE from imbalanced-learn + `class_weight='balanced'` in XGBoost
- **Primary metric:** AUC-ROC and Recall (not accuracy — dataset is 99.8% non-fraud)
- **SHAP output:** Export `top_shap_feature` and `shap_value` per transaction into `fraud_results.csv`
- **Export columns for BI:** `transaction_id, amount, hour, predicted_label, fraud_probability, top_shap_feature, shap_value`

---

## Tableau Public Dashboard Plan — "Fraud Intelligence Center"
- Precision/Recall curve (dual-axis + parameter threshold slider)
- Hourly fraud rate heatmap (LOD expression)
- SHAP feature importance bar (set action)
- Cumulative fraud caught / lift curve (table calculation)
- Fraud amount distribution (dual-axis histogram + boxplot)

## Power BI Report Plan — "Fraud Executive Dashboard"
- What-If parameter: risk threshold slider → live Precision/Recall DAX
- Key Influencers visual (AI-driven)
- Decomposition Tree: fraud → hour → amount range → top SHAP feature
- Smart Narrative auto-summary
- Drillthrough page: per-transaction SHAP breakdown

---

## Model Results (Actual)
| Model | Precision | Recall | F1 | AUC-ROC | AUC-PR |
|---|---|---|---|---|---|
| Logistic Regression | 0.0557 | 0.9184 | 0.1050 | 0.9706 | 0.7277 |
| Random Forest | 0.5321 | 0.8469 | 0.6535 | 0.9804 | 0.8281 |
| **XGBoost** | 0.1360 | 0.8980 | 0.2362 | 0.9794 | **0.8396** |

**Best model:** XGBoost (AUC-PR = 0.84, best for imbalanced data)
**Threshold tuning needed:** XGBoost precision is low at default 0.5 — evaluate.py will find optimal cutoff

## Progress Tracker
- [x] Data downloaded from Kaggle
- [x] preprocess.py complete
- [x] train.py complete (3 models — results above)
- [ ] 01_eda.ipynb complete
- [x] evaluate.py complete (AUC=0.9794, best F1 at threshold 0.9)
- [x] fraud_results.csv exported (2.1MB — ready for Tableau + Power BI)
- [x] confusion_matrix.png, roc_curve.png, shap_summary_bar.png, shap_summary_beeswarm.png saved
- [x] streamlit_app.py complete (runs on localhost:8501)
- [x] README.md complete (261 lines — update GitHub/LinkedIn/Live Demo URLs before publishing)
- [ ] GitHub repo created + pushed
- [ ] Streamlit Community Cloud deployed (get public URL)
- [ ] Tableau Public dashboard built + published
- [ ] Power BI report built + published
- [ ] README Live Demo URLs updated with all 3 links

---

## Notes for Claude Code
- Always save intermediate outputs to `data/` or `dashboards/`
- Use `pickle` to save models to `models/`
- Export `fraud_results.csv` with ALL columns listed above — needed for Tableau + Power BI
- Streamlit app must load `models/xgb_fraud_model.pkl` and `data/creditcard.csv` sample
- Write clean, commented code — this is a portfolio project
