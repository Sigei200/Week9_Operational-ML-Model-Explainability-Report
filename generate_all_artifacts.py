"""
Script to execute the complete Week 9 Operational ML workflow:
- Data Prep & Imbalance Analysis
- Leakage-Free Stratified 5-Fold Cross Validation
- Model Comparison: Random Forest, SMOTE-RF, XGBoost, SMOTE-XGBoost
- Operational Metrics (Precision, Recall, F1, ROC-AUC, PR-AUC)
- Cost-Sensitive Threshold Optimization
- K-Means Clustering for Operational Regimes
- SHAP Global & Local Explainability
- 2-Page Executive PDF Report generation
- Capstone Update documentation
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import StratifiedKFold, train_test_split, cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    precision_recall_curve, roc_curve, auc, f1_score, precision_score, recall_score
)
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
import xgboost as xgb
import shap

def run_pipeline():
    # Configure styling
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
    plt.rcParams['figure.dpi'] = 300
    plt.rcParams['axes.titlesize'] = 14
    plt.rcParams['axes.labelsize'] = 12

    os.makedirs('images', exist_ok=True)
    os.makedirs('data', exist_ok=True)

    print("="*70)
    print("STEP 1: DATA INGESTION & FEATURE ENGINEERING")
    print("="*70)

    data_file = 'data/ai4i2020.csv'
    if not os.path.exists(data_file):
        raise FileNotFoundError(f"Dataset not found at {data_file}")

    df_raw = pd.read_csv(data_file)
    print(f"Dataset Loaded: {df_raw.shape[0]} rows, {df_raw.shape[1]} columns")

    # Inspect class balance
    target_name = 'Machine failure' if 'Machine failure' in df_raw.columns else 'Machine_failure'
    failure_counts = df_raw[target_name].value_counts()
    failure_pcts = df_raw[target_name].value_counts(normalize=True) * 100
    print(f"Nominal Operations (0): {failure_counts[0]} ({failure_pcts[0]:.2f}%)")
    print(f"Machine Failures   (1): {failure_counts[1]} ({failure_pcts[1]:.2f}%)")
    print(f"Imbalance Ratio: {failure_counts[0] / failure_counts[1]:.1f} : 1")

    # Rename columns to remove square brackets for XGBoost compatibility
    df_raw = df_raw.rename(columns={
        'Air temperature [K]': 'Air_temperature_K',
        'Process temperature [K]': 'Process_temperature_K',
        'Rotational speed [rpm]': 'Rotational_speed_rpm',
        'Torque [Nm]': 'Torque_Nm',
        'Tool wear [min]': 'Tool_wear_min',
        'Machine failure': 'Machine_failure'
    })

    # Domain Feature Engineering
    df = df_raw.copy()
    # 1. Thermal Dissipation Gradient: Process Temp - Air Temp
    df['Temp_Diff_K'] = df['Process_temperature_K'] - df['Air_temperature_K']

    # 2. Mechanical Power (kW): Torque (Nm) * Speed (rad/s)
    df['Power_kW'] = (df['Torque_Nm'] * df['Rotational_speed_rpm'] * (2 * np.pi / 60)) / 1000.0

    # 3. Overstrain Index: Product of Tool Wear and Torque
    df['Overstrain_Index'] = df['Tool_wear_min'] * df['Torque_Nm']

    # 4. Temperature Ratio
    df['Temp_Ratio'] = df['Process_temperature_K'] / df['Air_temperature_K']

    # One-hot encode machine Type (L: Low quality 50%, M: Medium 30%, H: High 20%)
    if 'Type' in df.columns:
        df = pd.get_dummies(df, columns=['Type'], prefix='Type', drop_first=False)

    # Define feature columns and target
    feature_cols = [
        'Air_temperature_K', 'Process_temperature_K', 'Rotational_speed_rpm',
        'Torque_Nm', 'Tool_wear_min', 'Temp_Diff_K', 'Power_kW',
        'Overstrain_Index', 'Temp_Ratio', 'Type_H', 'Type_L', 'Type_M'
    ]
    target_col = 'Machine_failure'

    # Convert boolean columns to int
    for col in ['Type_H', 'Type_L', 'Type_M']:
        if col in df.columns:
            df[col] = df[col].astype(int)

    X = df[feature_cols]
    y = df[target_col]

    print(f"Engineered Features ({len(feature_cols)}): {feature_cols}")

    # Plot Class Imbalance
    fig, ax = plt.subplots(1, 2, figsize=(14, 5))
    colors = ['#2b5c8f', '#d9534f']
    ax[0].bar(['Nominal (0)', 'Machine Failure (1)'], failure_counts.values, color=colors, width=0.5, edgecolor='black', alpha=0.85)
    for i, v in enumerate(failure_counts.values):
        ax[0].text(i, v + 150, f"{v:,} ({failure_pcts.iloc[i]:.2f}%)", ha='center', fontweight='bold', fontsize=11)
    ax[0].set_title('Operational Class Distribution (Raw Counts)', fontsize=13, fontweight='bold')
    ax[0].set_ylabel('Number of Samples')
    ax[0].set_ylim(0, 11000)

    ax[1].pie(failure_counts.values, labels=['Nominal (96.61%)', 'Failure (3.39%)'], autopct='%1.2f%%',
              startangle=35, colors=colors, explode=(0, 0.15), shadow=True,
              textprops={'fontsize': 11, 'fontweight': 'bold'})
    ax[1].set_title('Operational Class Proportion (Extreme Imbalance)', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig('images/class_imbalance.png', dpi=300)
    plt.close()
    print("Saved: images/class_imbalance.png")

    print("\n" + "="*70)
    print("STEP 2: STRATIFIED TRAIN-TEST SPLIT (LEAKAGE PREVENTION)")
    print("="*70)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    print(f"Training Set: {X_train.shape[0]} samples | Failures: {y_train.sum()} ({y_train.mean()*100:.2f}%)")
    print(f"Test Set:     {X_test.shape[0]} samples | Failures: {y_test.sum()} ({y_test.mean()*100:.2f}%)")

    print("\n" + "="*70)
    print("STEP 3: 5-FOLD STRATIFIED CROSS-VALIDATION & MODEL BENCHMARKING")
    print("="*70)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scale_pos_weight_val = (len(y_train) - sum(y_train)) / sum(y_train)

    models = {
        'Random Forest (Balanced Weights)': RandomForestClassifier(
            n_estimators=150, max_depth=8, class_weight='balanced', random_state=42
        ),
        'SMOTE + Random Forest Pipeline': ImbPipeline([
            ('scaler', StandardScaler()),
            ('smote', SMOTE(random_state=42)),
            ('rf', RandomForestClassifier(n_estimators=150, max_depth=8, random_state=42))
        ]),
        'XGBoost (Scale_Pos_Weight)': xgb.XGBClassifier(
            n_estimators=150, max_depth=5, learning_rate=0.05,
            scale_pos_weight=scale_pos_weight_val, random_state=42,
            eval_metric='logloss'
        ),
        'SMOTE + XGBoost Pipeline': ImbPipeline([
            ('scaler', StandardScaler()),
            ('smote', SMOTE(random_state=42)),
            ('xgb', xgb.XGBClassifier(
                n_estimators=150, max_depth=5, learning_rate=0.05,
                random_state=42, eval_metric='logloss'
            ))
        ])
    }

    scoring = ['precision', 'recall', 'f1', 'roc_auc']
    cv_results = {}

    for name, model in models.items():
        print(f"Evaluating {name} with 5-Fold Stratified CV...")
        scores = cross_validate(model, X_train, y_train, cv=cv, scoring=scoring)
        cv_results[name] = {
            'Precision': float(np.mean(scores['test_precision'])),
            'Precision_std': float(np.std(scores['test_precision'])),
            'Recall': float(np.mean(scores['test_recall'])),
            'Recall_std': float(np.std(scores['test_recall'])),
            'F1-Score': float(np.mean(scores['test_f1'])),
            'F1_std': float(np.std(scores['test_f1'])),
            'ROC-AUC': float(np.mean(scores['test_roc_auc'])),
            'ROC_AUC_std': float(np.std(scores['test_roc_auc']))
        }
        print(f"  -> Recall: {cv_results[name]['Recall']:.4f} | Precision: {cv_results[name]['Precision']:.4f} | F1: {cv_results[name]['F1-Score']:.4f} | ROC-AUC: {cv_results[name]['ROC-AUC']:.4f}")

    cv_df = pd.DataFrame(cv_results).T
    print("\n--- 5-Fold Cross-Validation Benchmark Summary ---")
    print(cv_df[['Recall', 'Precision', 'F1-Score', 'ROC-AUC']])

    # Plot Model CV Comparison
    fig, ax = plt.subplots(figsize=(12, 6))
    metrics_plot = ['Recall', 'Precision', 'F1-Score', 'ROC-AUC']
    x_idx = np.arange(len(models))
    bar_width = 0.2
    palette = ['#e74c3c', '#3498db', '#2ecc71', '#9b59b6']

    for i, metric in enumerate(metrics_plot):
        means = [cv_results[m][metric] for m in models]
        std_key = metric.replace('-Score', '').replace('-', '_') + '_std'
        stds = [cv_results[m][std_key] for m in models]
        ax.bar(x_idx + i*bar_width, means, yerr=stds, width=bar_width, label=metric, color=palette[i], capsize=4, edgecolor='black', alpha=0.9)

    ax.set_xticks(x_idx + bar_width * 1.5)
    ax.set_xticklabels([m.replace(' Pipeline', '\nPipeline').replace(' (', '\n(') for m in models], fontsize=10, fontweight='bold')
    ax.set_ylabel('Score (Cross-Validation Mean)', fontsize=12)
    ax.set_ylim(0.45, 1.02)
    ax.set_title('Model Performance Across Operational Metrics (5-Fold Stratified CV)', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', frameon=True, fontsize=11)
    plt.tight_layout()
    plt.savefig('images/model_cv_comparison.png', dpi=300)
    plt.close()
    print("Saved: images/model_cv_comparison.png")

    print("\n" + "="*70)
    print("STEP 4: FINAL MODEL TRAINING & HOLDOUT TEST EVALUATION")
    print("="*70)

    # Train Production XGBoost and Random Forest
    xgb_model = xgb.XGBClassifier(
        n_estimators=150, max_depth=5, learning_rate=0.05,
        scale_pos_weight=scale_pos_weight_val, random_state=42,
        eval_metric='logloss'
    )
    xgb_model.fit(X_train, y_train)

    rf_model = RandomForestClassifier(
        n_estimators=150, max_depth=8, class_weight='balanced', random_state=42
    )
    rf_model.fit(X_train, y_train)

    # Predictions
    y_pred_xgb = xgb_model.predict(X_test)
    y_prob_xgb = xgb_model.predict_proba(X_test)[:, 1]

    y_pred_rf = rf_model.predict(X_test)
    y_prob_rf = rf_model.predict_proba(X_test)[:, 1]

    print("\n--- XGBoost Test Classification Report ---")
    print(classification_report(y_test, y_pred_xgb, target_names=['Nominal', 'Machine Failure'], digits=4))

    # Confusion Matrices
    cm_xgb = confusion_matrix(y_test, y_pred_xgb)
    cm_xgb_norm = confusion_matrix(y_test, y_pred_xgb, normalize='true')

    cm_rf = confusion_matrix(y_test, y_pred_rf)
    cm_rf_norm = confusion_matrix(y_test, y_pred_rf, normalize='true')

    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    sns.heatmap(cm_xgb, annot=True, fmt='d', cmap='Blues', ax=axes[0, 0], cbar=False,
                xticklabels=['Pred: Nominal', 'Pred: Failure'], yticklabels=['True: Nominal', 'True: Failure'])
    axes[0, 0].set_title('XGBoost - Confusion Matrix (Raw Counts)', fontsize=12, fontweight='bold')

    sns.heatmap(cm_xgb_norm, annot=True, fmt='.2%', cmap='Blues', ax=axes[0, 1], cbar=False,
                xticklabels=['Pred: Nominal', 'Pred: Failure'], yticklabels=['True: Nominal', 'True: Failure'])
    axes[0, 1].set_title('XGBoost - Normalized Confusion Matrix (Operational Recall)', fontsize=12, fontweight='bold')

    sns.heatmap(cm_rf, annot=True, fmt='d', cmap='Greens', ax=axes[1, 0], cbar=False,
                xticklabels=['Pred: Nominal', 'Pred: Failure'], yticklabels=['True: Nominal', 'True: Failure'])
    axes[1, 0].set_title('Random Forest - Confusion Matrix (Raw Counts)', fontsize=12, fontweight='bold')

    sns.heatmap(cm_rf_norm, annot=True, fmt='.2%', cmap='Greens', ax=axes[1, 1], cbar=False,
                xticklabels=['Pred: Nominal', 'Pred: Failure'], yticklabels=['True: Nominal', 'True: Failure'])
    axes[1, 1].set_title('Random Forest - Normalized Confusion Matrix (Operational Recall)', fontsize=12, fontweight='bold')

    plt.tight_layout()
    plt.savefig('images/confusion_matrices.png', dpi=300)
    plt.close()
    print("Saved: images/confusion_matrices.png")

    # ROC and PR Curves
    fpr_xgb, tpr_xgb, _ = roc_curve(y_test, y_prob_xgb)
    roc_auc_xgb_val = auc(fpr_xgb, tpr_xgb)

    fpr_rf, tpr_rf, _ = roc_curve(y_test, y_prob_rf)
    roc_auc_rf_val = auc(fpr_rf, tpr_rf)

    prec_xgb, rec_xgb, _ = precision_recall_curve(y_test, y_prob_xgb)
    pr_auc_xgb_val = auc(rec_xgb, prec_xgb)

    prec_rf, rec_rf, _ = precision_recall_curve(y_test, y_prob_rf)
    pr_auc_rf_val = auc(rec_rf, prec_rf)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    # ROC
    axes[0].plot(fpr_xgb, tpr_xgb, color='#e74c3c', lw=2.5, label=f'XGBoost (AUC = {roc_auc_xgb_val:.4f})')
    axes[0].plot(fpr_rf, tpr_rf, color='#2ecc71', lw=2.5, label=f'Random Forest (AUC = {roc_auc_rf_val:.4f})')
    axes[0].plot([0, 1], [0, 1], color='navy', lw=1.5, linestyle='--', label='Random Chance (AUC = 0.50)')
    axes[0].set_xlim([0.0, 1.0])
    axes[0].set_ylim([0.0, 1.05])
    axes[0].set_xlabel('False Positive Rate (1 - Specificity)')
    axes[0].set_ylabel('True Positive Rate (Recall / Sensitivity)')
    axes[0].set_title('Receiver Operating Characteristic (ROC) Curves', fontsize=13, fontweight='bold')
    axes[0].legend(loc='lower right', frameon=True)

    # PR Curve
    baseline_pr = y_test.sum() / len(y_test)
    axes[1].plot(rec_xgb, prec_xgb, color='#e74c3c', lw=2.5, label=f'XGBoost (PR-AUC = {pr_auc_xgb_val:.4f})')
    axes[1].plot(rec_rf, prec_rf, color='#2ecc71', lw=2.5, label=f'Random Forest (PR-AUC = {pr_auc_rf_val:.4f})')
    axes[1].plot([0, 1], [baseline_pr, baseline_pr], color='navy', lw=1.5, linestyle='--', label=f'No-Skill Baseline ({baseline_pr:.2%})')
    axes[1].set_xlim([0.0, 1.0])
    axes[1].set_ylim([0.0, 1.05])
    axes[1].set_xlabel('Recall (Detection Rate)')
    axes[1].set_ylabel('Precision (True Alarm Purity)')
    axes[1].set_title('Precision-Recall Curves for Rare Event Detection', fontsize=13, fontweight='bold')
    axes[1].legend(loc='upper right', frameon=True)

    plt.tight_layout()
    plt.savefig('images/roc_pr_curves.png', dpi=300)
    plt.close()
    print("Saved: images/roc_pr_curves.png")

    print("\n" + "="*70)
    print("STEP 5: COST-SENSITIVE DECISION THRESHOLD OPTIMIZATION")
    print("="*70)

    cost_fn = 15000
    cost_fp = 500
    cost_tp = 1200

    thresholds = np.linspace(0.05, 0.95, 100)
    total_costs = []
    f1_scores = []
    recalls = []
    precisions = []

    for t in thresholds:
        preds = (y_prob_xgb >= t).astype(int)
        cm = confusion_matrix(y_test, preds)
        tn, fp, fn, tp = cm.ravel()
        
        cost = (fn * cost_fn) + (fp * cost_fp) + (tp * cost_tp)
        total_costs.append(cost)
        f1_scores.append(f1_score(y_test, preds))
        recalls.append(recall_score(y_test, preds))
        precisions.append(precision_score(y_test, preds, zero_division=0))

    opt_idx = np.argmin(total_costs)
    optimal_threshold = thresholds[opt_idx]
    min_cost = total_costs[opt_idx]
    default_cost = total_costs[np.argmin(np.abs(thresholds - 0.50))]

    print(f"Default Threshold (0.50) Expected Cost: ${default_cost:,.2f}")
    print(f"Optimal Threshold ({optimal_threshold:.2f}) Expected Cost: ${min_cost:,.2f}")
    print(f"Financial Savings: ${default_cost - min_cost:,.2f} per 2,000 operational cycles ({((default_cost - min_cost)/default_cost)*100:.1f}% reduction)")

    fig, ax1 = plt.subplots(figsize=(12, 6))
    ax2 = ax1.twinx()

    line1 = ax1.plot(thresholds, np.array(total_costs)/1000.0, color='#d9534f', lw=3, label='Total Operational Cost ($k)')
    ax1.axvline(optimal_threshold, color='black', linestyle='--', lw=2, label=f'Optimal Threshold ({optimal_threshold:.2f})')
    ax1.axvline(0.50, color='gray', linestyle=':', lw=2, label='Default Threshold (0.50)')

    line2 = ax2.plot(thresholds, recalls, color='#3498db', lw=2, linestyle='-', label='Recall (Catch Rate)')
    line3 = ax2.plot(thresholds, precisions, color='#2ecc71', lw=2, linestyle='-', label='Precision')
    line4 = ax2.plot(thresholds, f1_scores, color='#9b59b6', lw=2, linestyle='--', label='F1-Score')

    ax1.set_xlabel('Classification Probability Decision Threshold', fontsize=12)
    ax1.set_ylabel('Total Operational Cost ($ in Thousands)', color='#d9534f', fontsize=12)
    ax2.set_ylabel('Operational Metric Score', fontsize=12)
    ax1.tick_params(axis='y', labelcolor='#d9534f')
    ax1.set_title('Cost-Sensitive Decision Threshold Tuning for Operational Safety', fontsize=14, fontweight='bold')

    lines = line1 + [ax1.get_lines()[1], ax1.get_lines()[2]] + line2 + line3 + line4
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='center right', frameon=True)
    plt.tight_layout()
    plt.savefig('images/cost_threshold_optimization.png', dpi=300)
    plt.close()
    print("Saved: images/cost_threshold_optimization.png")

    print("\n" + "="*70)
    print("STEP 6: UNSUPERVISED K-MEANS CLUSTERING (OPERATIONAL REGIMES)")
    print("="*70)

    telemetry_features = ['Air_temperature_K', 'Process_temperature_K', 'Rotational_speed_rpm', 'Torque_Nm', 'Tool_wear_min', 'Temp_Diff_K', 'Power_kW', 'Overstrain_Index']
    scaler_km = StandardScaler()
    X_km_scaled = scaler_km.fit_transform(df[telemetry_features])

    k_range = range(2, 7)
    inertias = []
    sil_scores = []

    for k in k_range:
        km_test = KMeans(n_clusters=k, random_state=42, n_init=10)
        km_test.fit(X_km_scaled)
        inertias.append(km_test.inertia_)
        sil_scores.append(silhouette_score(X_km_scaled, km_test.labels_))

    kmeans_final = KMeans(n_clusters=4, random_state=42, n_init=10)
    df['Cluster'] = kmeans_final.fit_predict(X_km_scaled)

    cluster_summary = df.groupby('Cluster')[telemetry_features + ['Machine_failure']].mean()
    cluster_summary['Sample_Count'] = df['Cluster'].value_counts()
    cluster_summary['Failure_Rate_%'] = df.groupby('Cluster')['Machine_failure'].mean() * 100

    print("\n--- Operational Cluster Profiles ---")
    print(cluster_summary[['Sample_Count', 'Failure_Rate_%', 'Power_kW', 'Torque_Nm', 'Temp_Diff_K', 'Tool_wear_min']])

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

    axes[0].plot(list(k_range), inertias, 'bo-', lw=2, markersize=8)
    axes[0].set_title('Elbow Method (Inertia)', fontsize=13, fontweight='bold')
    axes[0].set_xlabel('Number of Clusters (k)')
    axes[0].set_ylabel('Within-Cluster Sum of Squares')

    axes[1].plot(list(k_range), sil_scores, 'gs-', lw=2, markersize=8)
    axes[1].set_title('Silhouette Score by Cluster Count', fontsize=13, fontweight='bold')
    axes[1].set_xlabel('Number of Clusters (k)')
    axes[1].set_ylabel('Silhouette Coefficient')

    scatter_colors = ['#3498db', '#9b59b6', '#2ecc71', '#e67e22']
    for c_id in range(4):
        c_mask = df['Cluster'] == c_id
        axes[2].scatter(df.loc[c_mask, 'Rotational_speed_rpm'], df.loc[c_mask, 'Torque_Nm'],
                        c=scatter_colors[c_id], label=f'Cluster {c_id} ({cluster_summary.loc[c_id, "Failure_Rate_%"]:.1f}% Fail)',
                        alpha=0.4, s=20)

    fail_mask = df['Machine_failure'] == 1
    axes[2].scatter(df.loc[fail_mask, 'Rotational_speed_rpm'], df.loc[fail_mask, 'Torque_Nm'],
                    c='red', marker='x', s=60, linewidths=1.5, label='Actual Failures', alpha=0.9)

    axes[2].set_title('Operational Regimes: Torque vs Speed', fontsize=13, fontweight='bold')
    axes[2].set_xlabel('Rotational Speed [rpm]')
    axes[2].set_ylabel('Torque [Nm]')
    axes[2].legend(loc='upper right', frameon=True, fontsize=9)

    plt.tight_layout()
    plt.savefig('images/kmeans_clustering_analysis.png', dpi=300)
    plt.close()
    print("Saved: images/kmeans_clustering_analysis.png")

    print("\n" + "="*70)
    print("STEP 7: SHAP EXPLAINABILITY ENGINE (GLOBAL & LOCAL)")
    print("="*70)

    explainer = shap.TreeExplainer(xgb_model)
    shap_values = explainer(X_test)

    fig, ax = plt.subplots(figsize=(11, 7))
    shap.summary_plot(shap_values, X_test, feature_names=feature_cols, show=False)
    plt.title('SHAP Global Feature Impact on Machine Failure Risk', fontsize=14, fontweight='bold', pad=15)
    plt.tight_layout()
    plt.savefig('images/shap_global_summary.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved: images/shap_global_summary.png")

    mean_shap = np.abs(shap_values.values).mean(axis=0)
    shap_imp_df = pd.DataFrame({
        'Feature': feature_cols,
        'Mean_SHAP_Impact': mean_shap
    }).sort_values('Mean_SHAP_Impact', ascending=True)

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(shap_imp_df['Feature'], shap_imp_df['Mean_SHAP_Impact'], color='#2b5c8f', edgecolor='black', alpha=0.85)
    for bar in bars:
        width = bar.get_width()
        ax.text(width + 0.05, bar.get_y() + bar.get_height()/2, f'{width:.3f}',
                va='center', ha='left', fontsize=10, fontweight='bold')
    ax.set_title('Global Feature Importance Ranking (Mean |SHAP Value|)', fontsize=14, fontweight='bold')
    ax.set_xlabel('Mean |SHAP Value| (Average Absolute Impact on Log-Odds of Failure)', fontsize=11)
    ax.set_xlim(0, max(mean_shap)*1.15)
    plt.tight_layout()
    plt.savefig('images/shap_feature_importance_bar.png', dpi=300)
    plt.close()
    print("Saved: images/shap_feature_importance_bar.png")

    test_failures = np.where((y_test.values == 1) & (y_prob_xgb > 0.85))[0]
    sample_idx = int(test_failures[0])

    fig, ax = plt.subplots(figsize=(10, 6))
    shap.plots.waterfall(shap_values[sample_idx], show=False, max_display=8)
    plt.title(f'Local Root-Cause Waterfall Explanation for Critical Incident #{sample_idx}\n(Predicted Risk: {y_prob_xgb[sample_idx]:.1%})',
              fontsize=13, fontweight='bold', pad=15)
    plt.tight_layout()
    plt.savefig('images/shap_waterfall_case.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved: images/shap_waterfall_case.png")

    # Save summary metrics to json for notebook and PDF generators
    metrics_summary = {
        'total_samples': len(df),
        'failure_count': int(failure_counts[1]),
        'nominal_count': int(failure_counts[0]),
        'failure_pct': float(failure_pcts.iloc[1]),
        'test_samples': len(y_test),
        'test_failures': int(y_test.sum()),
        'xgb_roc_auc': float(roc_auc_xgb_val),
        'xgb_pr_auc': float(pr_auc_xgb_val),
        'rf_roc_auc': float(roc_auc_rf_val),
        'rf_pr_auc': float(pr_auc_rf_val),
        'optimal_threshold': float(optimal_threshold),
        'default_cost': float(default_cost),
        'min_cost': float(min_cost),
        'cost_savings': float(default_cost - min_cost),
        'top_features': shap_imp_df.tail(3)['Feature'].tolist(),
        'cv_results': cv_results
    }

    with open('data/metrics_summary.json', 'w') as f:
        json.dump(metrics_summary, f, indent=4)

    print("\n" + "="*70)
    print("METRICS SAVED TO data/metrics_summary.json")
    print("ALL DATA SCIENCE STEPS COMPLETED!")
    print("="*70)

if __name__ == '__main__':
    run_pipeline()
