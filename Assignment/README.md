# Week 9 Operational ML & Model Explainability (XAI)
## Industrial Predictive Maintenance & Telemetry Failure Detection Pipeline

[![Python 3.11](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.9-orange.svg)](https://scikit-learn.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-3.2-red.svg)](https://xgboost.readthedocs.io/)
[![SHAP](https://img.shields.io/badge/SHAP-0.51-brightgreen.svg)](https://shap.readthedocs.io/)
[![ReportLab](https://img.shields.io/badge/ReportLab-5.0-blueviolet.svg)](https://www.reportlab.com/)

An end-to-end production-grade machine learning pipeline, cost-sensitive threshold optimizer, unsupervised operational clustering, and explainable AI (SHAP) architecture designed for industrial predictive maintenance and telemetry failure prevention.

---

## 📁 Repository Structure

```
d:/EMTECH/PLP/Week 9/Assignment/
├── data/
│   ├── ai4i2020.csv                       # Benchmark telemetry dataset (10k records, 3.39% failure)
│   └── metrics_summary.json               # Serialized benchmark evaluation metrics
├── images/
│   ├── class_imbalance.png                # Operational class distribution (96.6% vs 3.39%)
│   ├── model_cv_comparison.png            # 5-Fold Stratified CV benchmark chart
│   ├── confusion_matrices.png             # Raw and normalized confusion matrices
│   ├── roc_pr_curves.png                  # ROC-AUC and Precision-Recall curves
│   ├── cost_threshold_optimization.png    # Financial cost curve vs decision threshold
│   ├── kmeans_clustering_analysis.png     # Elbow, Silhouette, and operational regimes scatter
│   ├── shap_global_summary.png            # Global SHAP beeswarm feature impact plot
│   ├── shap_feature_importance_bar.png    # Mean |SHAP| enterprise feature ranking
│   └── shap_waterfall_case.png            # Local root-cause waterfall incident explanation
├── week9_operational_ml.ipynb             # [Part A] Fully executed Jupyter Notebook
├── Week9_Model_Explainer_Report.pdf       # [Part B] Professional 2-page executive PDF report
├── Week9_Model_Explainer_Student.pdf      # [Part B] PDF Deliverable
├── Week9_Model_Explainer.md               # [Part B] Markdown companion of the Model Explainer
├── capstone_week9_update.md               # [Part C] Capstone ML milestone update
├── generate_all_artifacts.py              # Automated data pipeline & modeling execution script
├── build_notebook.py                      # Programmatic Jupyter Notebook compiler
├── generate_pdf_report.py                 # ReportLab 2-page PDF generator
└── README.md                              # Master submission documentation
```

---

## 🚀 Key Deliverables Overview

### 1. Part A: Technical Coding Challenge (`week9_operational_ml.ipynb`)
* **Problem Definition:** Industrial telemetry failure prediction on the UCI AI4I 2020 dataset (10,000 operational cycles, measuring Air/Process temperatures, rotational speed, torque, tool wear, and machine quality types).
* **Imbalance Mitigation:** Solves the **96.61% Nominal vs. 3.39% Failure** extreme skew using:
  * Leakage-free `imblearn.pipeline.Pipeline` with `SMOTE`.
  * Algorithmic loss weighting (`scale_pos_weight = 28.5`).
* **Model Benchmark (5-Fold Stratified Cross-Validation):**

| Model Configuration | Recall (Catch Rate) | Precision | F1-Score | ROC-AUC |
| :--- | :---: | :---: | :---: | :---: |
| **Random Forest (Balanced Weights)** | 85.23% | 61.21% | 0.7122 | 0.9817 |
| **SMOTE + Random Forest Pipeline** | **87.80%** | 49.68% | 0.6343 | 0.9793 |
| **XGBoost (Scale_Pos_Weight)** | 84.13% | **66.58%** | **0.7428** | 0.9789 |
| **SMOTE + XGBoost Pipeline** | 87.43% | 48.65% | 0.6249 | 0.9780 |

* **Holdout Test Evaluation (2,000 Unseen Cycles):**
  * **Recall:** $86.76\%$ (Catches 59 out of 68 actual impending breakdowns).
  * **ROC-AUC:** $0.9806$ | **PR-AUC:** $0.7314$ (vs $3.4\%$ random baseline).
* **Cost-Sensitive Threshold Tuning:** Sweeps probability thresholds under asymmetric costs ($C_{FN} = \$15,000 \gg C_{FP} = \$500$). Tuning threshold to $t = 0.20$ saves over **$\$210,000+$** in annual downtime losses.
* **Unsupervised Clustering (K-Means):** Discovers 4 distinct operational stress regimes (Safe Normal, High-Speed Low-Torque, Heavy-Torque High-Power, High-Wear Critical Regime).
* **SHAP Explainability:** Computes TreeSHAP for global beeswarm impact, feature importance ranking, and local incident waterfall breakdowns.

---

### 2. Part B: Soft Skills Deliverable (`Week9_Model_Explainer_Report.pdf` & `.md`)
* A polished, 2-page executive report tailored for plant managers and operations directors.
* **The Analogy:** Translates complex ML concepts into the familiar framing of a **"24/7 Digital Chief Inspector."**
* **The Drivers:** Explains the Top 3 physical failure drivers in plain English:
  1. *Overstrain Index* ($\text{Tool Wear} \times \text{Torque}$)
  2. *Thermal Dissipation Gradient* ($T_{\text{Process}} - T_{\text{Air}}$)
  3. *Mechanical Power Dissipation* ($\text{kW}$)
* **Incident Walkthrough:** Complete breakdown of high-risk Incident #196 using local SHAP waterfall attribution.
* **Standard Operating Procedure (SOP):** 4-step protocol including a **5-Minute Telemetry Stabilization Buffer** to filter startup transients.

---

### 3. Part C: Capstone Progress Check (`capstone_week9_update.md`)
* Integrated milestone progress report for the **Healthcare Supply-Chain Intelligence Platform** (`D:\EMTECH\PLP\Capstone\Capstone_project`).
* Answers all 3 required milestone questions:
  1. *ML Algorithm Choice:* XGBoost + Random Forest for tabular non-linear interaction modeling and fast $\mathcal{O}(TLD^2)$ TreeSHAP inference.
  2. *Imbalance Handling:* Stratified CV, SMOTE training pipelines, cost-sensitive loss (`scale_pos_weight`), and PR-AUC/Recall prioritization.
  3. *Surprising SHAP Insight:* Compound interaction features (e.g. *Supplier Lead-Time Volatility* and *Thermal Differential Gradients*) exhibit over $3\times$ higher predictive power than static volumetric balances or raw absolute temperatures.

---

## ⚙️ How to Reproduce & Run

### 1. Environment Setup
```bash
# Clone the repository
git clone <your-repo-link>
cd Assignment

# Initialize virtual environment and install dependencies
uv venv .venv --python 3.11
uv pip install -r requirements.txt
```

### 2. Run the Full Automated Pipeline
```bash
# Run data pipeline, model training, and generate all high-res plots
python generate_all_artifacts.py

# Build and execute the Jupyter Notebook
python build_notebook.py
python -m nbconvert --to notebook --execute week9_operational_ml.ipynb --inplace

# Compile the 2-Page Executive PDF Report
python generate_pdf_report.py
```

---

## 📊 Evaluation Rubric Compliance Checklist

| Rubric Criteria | Weight | Compliance Summary |
| :--- | :---: | :--- |
| **ML Workflow Rigor** | 30% | Leakage-free `imblearn` pipeline, 5-Fold Stratified CV, SMOTE & Class Weighting, feature engineering. |
| **Model Performance & Validation** | 25% | Multi-model benchmark (RF, SMOTE-RF, XGBoost, SMOTE-XGB), Confusion Matrix, ROC-AUC, PR-AUC, Threshold Optimization. |
| **Explainability Implementation** | 20% | SHAP TreeExplainer, Beeswarm Summary Plot, Mean \|SHAP\| Bar Chart, Local Waterfall Incident Attribution. |
| **Clarity & Trust-Building** | 15% | 2-Page PDF Explainer, shop-floor analogies, Top 3 drivers in plain English, 5-min stabilization buffer SOP. |
| **Code Docs & Capstone Progress** | 10% | Fully commented code cells, markdown equations, structured `capstone_week9_update.md`. |

