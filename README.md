# Telecom Customer Churn Prediction

> End-to-end machine learning framework for predicting and segmenting customer churn in telecommunications — combining gradient boosting, SMOTE-based class balancing, and behavioral clustering to deliver actionable retention insights.

---

## Overview

Customer churn costs telecom companies significantly more than retention. This project builds a production-grade churn prediction pipeline that identifies high-risk customers before they leave, enabling targeted, cost-efficient retention campaigns.

Using a multi-source IBM Cognos telecom dataset (~7,000 customers, 7 merged tables), we benchmark 7 classification algorithms across 28 configurations and develop an interpretable customer segmentation framework — achieving a **top-decile lift of 2.91** on the final model.

**Key result:** CatBoost with SMOTE achieves **0.93 CV ROC-AUC** and concentrates customers with a 77.3% churn rate in the top 10% of predicted risk, versus a 26.5% baseline.

---

## Results

### Model Performance — Top Configurations

| Rank | Model | Configuration | CV ROC-AUC | Test ROC-AUC | Test PR-AUC |
|------|-------|--------------|------------|--------------|-------------|
| 1 | **CatBoost** | Baseline + SMOTE | **0.934 ± 0.004** | 0.845 | 0.653 |
| 2 | LightGBM | Baseline + SMOTE | 0.932 ± 0.002 | 0.844 | 0.650 |
| 3 | XGBoost | Baseline + SMOTE | 0.930 ± 0.004 | 0.829 | 0.630 |
| 4 | Random Forest | Baseline + SMOTE | 0.928 ± 0.007 | 0.837 | 0.616 |
| — | Logistic Regression | Best config | 0.864 | — | — |

### Tuned Model — Final Metrics

| Metric | Value |
|--------|-------|
| CV ROC-AUC | 0.9399 |
| Test ROC-AUC | 0.8328 |
| Test PR-AUC | 0.6368 |
| Balanced Accuracy | 0.7306 |
| Matthews Correlation Coefficient | 0.449 |
| Brier Score | 0.158 |
| **Top-Decile Lift** | **2.91×** |

### Customer Segmentation

| Segment | Size | Median Tenure | Churn Risk | Profile |
|---------|------|--------------|------------|---------|
| Established Customers | 60% | ~15 months | Moderate | Stable revenue base; upsell candidates |
| New / At-Risk Customers | 40% | ~2 months | High (60–85) | First-90-day critical window; needs onboarding |

Silhouette score improved from **0.129 → 0.638** (+393%) after behavioral feature engineering and PCA.

---

## Tech Stack

| Component | Details |
|-----------|---------|
| Language | Python 3 |
| ML Models | CatBoost · LightGBM · XGBoost · Random Forest · Gradient Boosting · Logistic Regression · SVM |
| Imbalance Handling | SMOTE (imbalanced-learn) |
| Clustering | K-Means · Agglomerative (Ward) · GMM |
| Dimensionality Reduction | PCA (8 components, 95% variance explained) |
| Evaluation | ROC-AUC · PR-AUC · Balanced Accuracy · MCC · Top-Decile Lift · Brier Score |
| Statistical Testing | Kruskal-Wallis H-test |
| Data Processing | Pandas · NumPy · Scikit-Learn |
| Visualization | Matplotlib · Seaborn |

---

## Key Findings

**SMOTE is critical for tree-based models.** Ensemble methods gained 8–11% in CV ROC-AUC with synthetic oversampling (CatBoost +8.7%, XGBoost +10.6%), while Logistic Regression changed by only 0.7% — confirming that decision trees require sufficient minority-class density for meaningful splits, whereas linear models handle imbalance through class weighting.

**Gradient boosting dominates.** The gap between the best boosting method (CatBoost: 0.934) and best linear model (Logistic Regression: 0.864) quantifies the value of sequential error correction and automatic interaction modeling for tabular churn data.

**Clustering features don't help prediction.** Despite producing interpretable segments, adding cluster labels to the supervised feature set degraded performance across all models (CatBoost: −1.2%, XGBoost: −1.4%). The signals captured by clustering are already present in the raw features — but the segments remain valuable for designing differentiated retention strategies.

**Payment method and CLTV are the strongest churn predictors.** Engineered features (`charges_per_tenure`, `avg_monthly_charges`) ranked in the top 5, validating domain-informed feature construction even for models that learn interactions automatically.

**Feature engineering unlocked cluster structure.** The original feature space showed near-zero cluster signal (silhouette < 0.25). After constructing behavioral features (`services_count`, `charge_ratio`, `tenure_category`) and applying PCA, Hierarchical clustering achieved 0.638 silhouette — a 5× improvement.

---

## How to Run

```bash
# Clone the repo
git clone https://github.com/nikoskakonas/telecom-churn-prediction.git
cd telecom-churn-prediction

# Install dependencies
pip install -r requirements.txt

# Step 1: Preprocess and merge the 7 raw datasets
python preprocess.py

# Step 2: Run supervised learning (28-config comparison + tuning)
python supervised.py

# Step 3: Run clustering pipeline
python clustering_supervised.py

# Step 4: Run SMOTE + clustering combined pipeline
python merged_smote_clustering.py
```

---

## Project Structure

```
telecom-churn-prediction/
├── data/
│   ├── raw_data/                        # 7 IBM Cognos telecom datasets (.xlsx)
│   ├── merged_telco_preprocessed.csv    # Integrated dataset for supervised learning
│   ├── unsupervised_telco_preprocessed.csv
│   └── clustered_data_improved.csv      # Cluster assignments from best model
├── images/
│   └── final_report/                    # ROC curves, confusion matrices, feature importance
├── preprocess.py                        # Data merging and cleaning pipeline
├── supervised.py                        # 28-config model comparison + CatBoost tuning
├── clustering_supervised.py             # Clustering + supervised learning workflow
├── merged_smote_clustering.py           # SMOTE + clustering combined pipeline
├── model_comparison_results.csv         # Full results table
└── requirements.txt
```

---

## Methodology

### Data Pipeline
Seven IBM Cognos telecom tables (churn status, demographics, services, geography, population) were merged on `customer_id` and `zip_code`. Preprocessing included median imputation for numeric features, explicit "missing" categories for categorical variables, and behavioral feature engineering (`avg_monthly_charges`, `charges_per_tenure`, `services_count`).

### Supervised Learning
- **28 configurations**: 7 algorithms × 2 feature sets (baseline / cluster-enhanced) × 2 sampling strategies (raw / SMOTE)
- **Evaluation**: 5-fold stratified cross-validation + held-out test set
- **Selection**: Best CV ROC-AUC configuration (CatBoost + SMOTE) → GridSearchCV over 144 hyperparameter combinations
- **Metrics**: ROC-AUC, PR-AUC, Balanced Accuracy, MCC, Top-Decile Lift, Brier Score

### Unsupervised Learning
- **Baseline**: Raw features → silhouette < 0.25, no meaningful structure
- **Improved**: Feature engineering → VarianceThreshold filtering → RobustScaler → PCA (8 components, 95% variance) → Hierarchical clustering (Ward linkage)
- **Result**: Silhouette 0.638, two interpretable lifecycle segments

---

## Business Impact

A model with 2.91× top-decile lift means retention campaigns targeting the top 10% of predicted churners reach customers with a **77.3% churn rate** — nearly 3× the 26.5% base rate. This enables significant reduction in wasted spend on low-risk customers and more efficient allocation of retention offers, contract renegotiations, and support resources.

---

## Contributors

- **Nikolaos Kakonas** — Supervised & Unsupervised Learning Methods
- Andrew Park — Unsupervised Learning Models, Results
- Rishikesh Donthula — CatBoost Implementation, Visualizations, Metrics
- Saehee Eom — Unsupervised Learning, SMOTE
- Tejas Khandwekar — Supervised Learning, Data Preprocessing
