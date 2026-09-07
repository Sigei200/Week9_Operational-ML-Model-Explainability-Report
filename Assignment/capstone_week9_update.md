# Capstone Project Progress Check — Week 9 Milestone
**Project Title:** Healthcare Supply-Chain Intelligence Platform (Kenya Public Health Network)  
**Fellowship Module:** Week 9 — Operational Machine Learning & Explainability (XAI)  
**Repository Location:** `D:\EMTECH\PLP\Capstone\Capstone_project`

---

## 1. Which ML Algorithm Did You Choose and Why?

For the core operational prediction engines in our Capstone (specifically **Stockout Risk Forecasting**, **Batch Expiry Prediction**, and **Telemetry Breakdown Alerting**), we selected **XGBoost (Extreme Gradient Boosting)** complemented by **Random Forest** ensembles.

### Technical & Operational Justification:
* **Handling Tabular Non-Linearities:** Healthcare supply chain telemetry and operational logs are predominantly structured tabular data. XGBoost excels at capturing non-linear interactions (such as the compounding risk of high facility demand velocity paired with supplier lead-time variance) that linear models fail to capture.
* **Algorithmic Efficiency & Regularization:** XGBoost provides built-in L1 ($\alpha$) and L2 ($\lambda$) regularization, preventing leaf-weight overfitting on noisy historical records while maintaining ultra-fast inference suitable for real-time dashboard alerting.
* **Seamless Explainability Integration:** Tree-based boosting algorithms natively support exact **TreeSHAP** computation in $\mathcal{O}(TLD^2)$ time, enabling our platform to generate instant, mathematically consistent root-cause breakdowns for every automated alert dispatched to county health directors.

---

## 2. How Did You Handle Class Imbalance (If Applicable)?

### Operational Context:
In our public health supply chain, acute critical stockouts and severe supplier delivery disruptions occur in approximately **$3.39\% - 4.2\%$** of operational records. If evaluated with naive accuracy, a model predicting "No Stockout" for every facility achieves $>96\%$ accuracy while leaving rural health clinics without life-saving commodities.

### Multi-Tiered Imbalance Strategy Implemented:
1. **Leakage-Free Pipeline Architecture:** Implemented `imblearn.pipeline.Pipeline` integrating `StandardScaler` and `SMOTE(random_state=42)`. Resampling is executed strictly within training folds during cross-validation, preventing test data distribution from contaminating model training.
2. **Cost-Sensitive Loss Weighting (`scale_pos_weight`):** Configured XGBoost with $\text{scale\_pos\_weight} \approx 28.5$ ($N_{\text{majority}} / N_{\text{minority}}$). This penalizes the objective function heavily for missed minority events (False Negatives), aligning model loss with the real-world healthcare impact of stockouts.
3. **Operational Metric Prioritization:** Replaced accuracy with **Recall (Sensitivity)**, **Precision**, **F1-Score**, and **Precision-Recall Area Under Curve (PR-AUC)**.
4. **Decision Threshold Tuning:** Tuned the probability decision threshold from the default $0.50$ down to $0.15 - 0.25$, maximizing the detection of high-risk events (achieving $>86\%$ holdout recall) while keeping precautionary false alarm rates minimal.

---

## 3. What Is One Insight from Your Feature Importance Analysis That Surprised You?

### 💡 The Surprising Finding:
Before conducting the SHAP (SHapley Additive exPlanations) analysis, our domain assumption was that **raw absolute stock volume** (or raw temperature in telemetry equipment) would serve as the primary indicator of failure/stockout risk.

### The SHAP Revelation:
SHAP global summary and waterfall plots revealed that **compound interaction dynamics** possessed vastly greater predictive power than static volumetric metrics:

1. **The Lead-Time / Consumption Interaction Overrides Raw Volume:** A facility holding moderate inventory had a dramatically higher risk of stockout if the **variance in supplier delivery lead time** fluctuated, compared to a facility with low inventory but consistent 2-day delivery cycles.
2. **Gradient Differential vs. Absolute Values:** In hardware telemetry, the **Thermal Dissipation Gradient ($\Delta T = T_{\text{Process}} - T_{\text{Air}}$)** and the **Overstrain Fatigue Index ($\text{Tool Wear} \times \text{Torque}$)** drove over **$84\%$** of model log-odds predictions, whereas raw absolute temperature alone ranked near the bottom of feature importance.

### Practical Impact on Capstone Development:
This insight directly shaped our feature engineering pipeline in `analytics_module/models/predictive.py`, prompting us to engineer dynamic ratio features (e.g., *Days of Stock Velocity*, *Supplier Lead Time Volatility Index*, and *Redistribution Value Density*) rather than relying purely on static warehouse balances.

---

## 4. Summary of Capstone ML Artifacts Delivered
* **Predictive Pipeline:** Integrated 5-fold cross-validated XGBoost and Random Forest classifiers.
* **Explainability Suite:** Live SHAP waterfall and summary charts generated for every alert.
* **Allocation Optimizer:** Network redistribution matching engine optimizing inter-facility transfers before triggering expensive emergency procurements.

