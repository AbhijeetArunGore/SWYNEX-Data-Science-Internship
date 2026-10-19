"""SWYNEX Modeling Artifacts & Figures Generator (Task 3).

Trains the baseline DummyClassifier and LogisticRegression models, computes
evaluation metrics on a patient-grouped test partition, exports machine-readable
metrics to reports/modeling/metrics.json, and generates 5 publication-quality
visualizations in reports/modeling/figures/.
"""
import os
import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.metrics import precision_recall_curve, roc_curve

from src.modeling import (
    build_logistic_regression_pipeline,
    create_patient_grouped_split,
    evaluate_model,
    prepare_feature_matrix,
    save_metrics_to_json
)

# Styling configuration
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams['font.sans-serif'] = 'Segoe UI'
plt.rcParams['axes.edgecolor'] = '#cccccc'
plt.rcParams['axes.linewidth'] = 0.8
plt.rcParams['figure.dpi'] = 300

DATA_PATH = os.path.join('data', 'processed', 'diabetic_data_clean.csv')
REPORTS_DIR = os.path.join('reports', 'modeling')
FIGURES_DIR = os.path.join(REPORTS_DIR, 'figures')


def generate_target_distribution_plot(y: pd.Series, output_dir: str):
    """Figure 1: Binary Target Class Balance."""
    counts = y.value_counts()
    pcts = (counts / len(y)) * 100

    plt.figure(figsize=(8, 5))
    bars = plt.bar(
        ['Not Readmitted <30 Days (0)', 'Readmitted <30 Days (1)'],
        [counts[0], counts[1]],
        color=['#2b5c8f', '#c0392b'],
        width=0.45,
        edgecolor='black',
        linewidth=0.6
    )
    plt.title('Binary Target Distribution (30-Day Early Readmission Benchmark)', fontsize=12, fontweight='bold', pad=12)
    plt.xlabel('Clinical Classification Cohort', fontsize=11, fontweight='semibold')
    plt.ylabel('Encounter Volume', fontsize=11, fontweight='semibold')
    plt.ylim(0, 105000)

    for bar, count, pct in zip(bars, [counts[0], counts[1]], [pcts[0], pcts[1]]):
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2.0, yval + 1800, f'{count:,}\n({pct:.2f}%)', ha='center', va='bottom', fontsize=10, fontweight='bold')

    plt.tight_layout()
    out_path = os.path.join(output_dir, '01_target_class_distribution.png')
    plt.savefig(out_path, bbox_inches='tight')
    plt.close()
    print(f"Generated: {out_path}")


def generate_confusion_matrix_plot(dummy_metrics: dict, lr_metrics: dict, output_dir: str):
    """Figure 2: Confusion Matrix Comparison."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    cm_dummy = np.array(dummy_metrics['confusion_matrix'])
    cm_lr = np.array(lr_metrics['confusion_matrix'])

    sns.heatmap(cm_dummy, annot=True, fmt=',d', cmap='Blues', cbar=False,
                xticklabels=['Pred 0 (No)', 'Pred 1 (<30)'],
                yticklabels=['True 0 (No)', 'True 1 (<30)'],
                ax=axes[0], annot_kws={'size': 12, 'weight': 'bold'})
    axes[0].set_title('A: Baseline DummyClassifier (Majority Class)\nRecall = 0.0% (Misses All Readmissions)', fontsize=11, fontweight='bold')
    axes[0].set_xlabel('Predicted Label', fontsize=10, fontweight='semibold')
    axes[0].set_ylabel('True Label', fontsize=10, fontweight='semibold')

    sns.heatmap(cm_lr, annot=True, fmt=',d', cmap='Blues', cbar=False,
                xticklabels=['Pred 0 (No)', 'Pred 1 (<30)'],
                yticklabels=['True 0 (No)', 'True 1 (<30)'],
                ax=axes[1], annot_kws={'size': 12, 'weight': 'bold'})
    axes[1].set_title(f"B: Balanced Logistic Regression\nRecall = {lr_metrics['recall']*100:.1f}% ({lr_metrics['true_positives']:,} Detected)", fontsize=11, fontweight='bold')
    axes[1].set_xlabel('Predicted Label', fontsize=10, fontweight='semibold')
    axes[1].set_ylabel('True Label', fontsize=10, fontweight='semibold')

    plt.suptitle('Held-Out Test Set Confusion Matrix Comparison (N=20,093)', fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()
    out_path = os.path.join(output_dir, '02_confusion_matrix_comparison.png')
    plt.savefig(out_path, bbox_inches='tight')
    plt.close()
    print(f"Generated: {out_path}")


def generate_metrics_comparison_plot(dummy_metrics: dict, lr_metrics: dict, output_dir: str):
    """Figure 3: Metric Comparison Bar Chart."""
    metric_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC', 'PR-AUC']
    dummy_vals = [
        dummy_metrics['accuracy'],
        dummy_metrics['precision'],
        dummy_metrics['recall'],
        dummy_metrics['f1_score'],
        dummy_metrics['roc_auc'],
        dummy_metrics['pr_auc']
    ]
    lr_vals = [
        lr_metrics['accuracy'],
        lr_metrics['precision'],
        lr_metrics['recall'],
        lr_metrics['f1_score'],
        lr_metrics['roc_auc'],
        lr_metrics['pr_auc']
    ]

    x = np.arange(len(metric_names))
    width = 0.35

    plt.figure(figsize=(11, 5.5))
    bars1 = plt.bar(x - width/2, dummy_vals, width, label='Dummy Baseline (Most Frequent)', color='#7f8c8d', edgecolor='black', linewidth=0.5)
    bars2 = plt.bar(x + width/2, lr_vals, width, label='Balanced Logistic Regression', color='#2b5c8f', edgecolor='black', linewidth=0.5)

    plt.title('Performance Comparison: Dummy Baseline vs. Logistic Regression', fontsize=12, fontweight='bold', pad=12)
    plt.xlabel('Evaluation Metric', fontsize=11, fontweight='semibold')
    plt.ylabel('Score (0.0 to 1.0)', fontsize=11, fontweight='semibold')
    plt.xticks(x, metric_names, fontsize=10, fontweight='semibold')
    plt.ylim(0, 1.05)
    plt.legend(frameon=True, loc='upper right')

    for bar in bars1:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, yval + 0.02, f'{yval:.3f}', ha='center', va='bottom', fontsize=8.5, color='#555555')

    for bar in bars2:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, yval + 0.02, f'{yval:.3f}', ha='center', va='bottom', fontsize=8.5, fontweight='bold', color='#1a3a5a')

    plt.tight_layout()
    out_path = os.path.join(output_dir, '03_model_metrics_comparison.png')
    plt.savefig(out_path, bbox_inches='tight')
    plt.close()
    print(f"Generated: {out_path}")


def generate_roc_and_pr_curves_plot(y_test: pd.Series, y_prob: np.ndarray, output_dir: str):
    """Figure 4: Precision-Recall Curve and ROC Curve."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = roc_auc_score(y_test, y_prob)
    axes[0].plot(fpr, tpr, color='#2b5c8f', linewidth=2.5, label=f'Logistic Regression (AUC = {roc_auc:.4f})')
    axes[0].plot([0, 1], [0, 1], color='#7f8c8d', linestyle='--', linewidth=1.5, label='Random Guess (AUC = 0.5000)')
    axes[0].set_title('A: Receiver Operating Characteristic (ROC) Curve', fontsize=11, fontweight='bold')
    axes[0].set_xlabel('False Positive Rate (1 - Specificity)', fontsize=10, fontweight='semibold')
    axes[0].set_ylabel('True Positive Rate (Recall)', fontsize=10, fontweight='semibold')
    axes[0].legend(loc='lower right', frameon=True)

    # PR Curve
    prec, rec, _ = precision_recall_curve(y_test, y_prob)
    pr_auc = average_precision_score(y_test, y_prob)
    base_rate = float(y_test.mean())
    axes[1].plot(rec, prec, color='#c0392b', linewidth=2.5, label=f'Logistic Regression (PR-AUC = {pr_auc:.4f})')
    axes[1].axhline(y=base_rate, color='#7f8c8d', linestyle='--', linewidth=1.5, label=f'Baseline Positive Rate ({base_rate:.4f})')
    axes[1].set_title('B: Precision-Recall (PR) Curve', fontsize=11, fontweight='bold')
    axes[1].set_xlabel('Recall (Sensitivity)', fontsize=10, fontweight='semibold')
    axes[1].set_ylabel('Precision (Positive Predictive Value)', fontsize=10, fontweight='semibold')
    axes[1].legend(loc='upper right', frameon=True)

    plt.suptitle('Held-Out Discrimination Curves for 30-Day Readmission', fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()
    out_path = os.path.join(output_dir, '04_precision_recall_and_roc_curves.png')
    plt.savefig(out_path, bbox_inches='tight')
    plt.close()
    print(f"Generated: {out_path}")


def generate_top_coefficients_plot(pipe, numerical_cols: list, categorical_cols: list, output_dir: str):
    """Figure 5: Top Positive & Negative Model Coefficients."""
    cat_encoder = pipe.named_steps['prep'].named_transformers_['cat']
    cat_feature_names = cat_encoder.get_feature_names_out(categorical_cols)
    all_feature_names = numerical_cols + list(cat_feature_names)

    coefs = pipe.named_steps['clf'].coef_[0]
    coef_df = pd.DataFrame({'feature': all_feature_names, 'coefficient': coefs})

    top_pos = coef_df.sort_values(by='coefficient', ascending=False).head(8)
    top_neg = coef_df.sort_values(by='coefficient', ascending=True).head(8)
    plot_df = pd.concat([top_neg, top_pos]).sort_values(by='coefficient', ascending=True)

    # Clean label names for display
    plot_df['display_name'] = plot_df['feature'].apply(lambda x: x.replace('medical_specialty_', 'Spec: ').replace('admission_type_desc_', 'Adm: ').replace('payer_code_', 'Payer: '))

    plt.figure(figsize=(10, 6.5))
    colors = ['#27ae60' if c < 0 else '#c0392b' for c in plot_df['coefficient']]
    bars = plt.barh(plot_df['display_name'], plot_df['coefficient'], color=colors, edgecolor='black', linewidth=0.5)

    plt.title('Top Positive and Negative Logistic Regression Coefficients\n(Predicting 30-Day Hospital Readmission)', fontsize=12, fontweight='bold', pad=12)
    plt.xlabel('Standardized Log-Odds Coefficient', fontsize=11, fontweight='semibold')
    plt.ylabel('Clinical Feature', fontsize=11, fontweight='semibold')
    plt.axvline(x=0, color='black', linestyle='-', linewidth=0.8)

    for bar in bars:
        w = bar.get_width()
        ha = 'left' if w >= 0 else 'right'
        offset = 0.03 if w >= 0 else -0.03
        plt.text(w + offset, bar.get_y() + bar.get_height()/2.0, f'{w:.2f}', va='center', ha=ha, fontsize=8.5, fontweight='bold')

    plt.tight_layout()
    out_path = os.path.join(output_dir, '05_top_feature_coefficients.png')
    plt.savefig(out_path, bbox_inches='tight')
    plt.close()
    print(f"Generated: {out_path}")


def main():
    os.makedirs(FIGURES_DIR, exist_ok=True)
    df = pd.read_csv(DATA_PATH)
    print(f"Loaded dataset: {df.shape[0]:,} rows x {df.shape[1]} columns")

    X, y, groups, num_cols, cat_cols = prepare_feature_matrix(df, exclude_leakage=True)
    print(f"Feature matrix: {X.shape[1]} features ({len(num_cols)} numerical, {len(cat_cols)} categorical)")

    X_train, X_test, y_train, y_test = create_patient_grouped_split(X, y, groups, test_size=0.2, random_state=42)
    print(f"Train samples: {len(X_train):,}, Test samples: {len(X_test):,}")

    # 1. Dummy Classifier
    dummy = DummyClassifier(strategy='most_frequent')
    dummy.fit(X_train, y_train)
    dummy_metrics = evaluate_model(dummy, X_test, y_test)
    dummy_metrics['model_name'] = 'DummyClassifier (Most Frequent)'

    # 2. Logistic Regression
    lr_pipe = build_logistic_regression_pipeline(num_cols, cat_cols, class_weight='balanced', random_state=42)
    lr_pipe.fit(X_train, y_train)
    lr_metrics = evaluate_model(lr_pipe, X_test, y_test)
    lr_metrics['model_name'] = 'LogisticRegression (Balanced Weights)'
    y_prob_lr = lr_pipe.predict_proba(X_test)[:, 1]

    # Save metrics JSON
    combined_metrics = {
        'dataset_summary': {
            'total_encounters': len(df),
            'train_encounters': len(X_train),
            'test_encounters': len(X_test),
            'features_used': X.shape[1],
            'test_positive_base_rate': round(float(y_test.mean()), 4)
        },
        'dummy_classifier': dummy_metrics,
        'logistic_regression': lr_metrics
    }
    json_path = os.path.join(REPORTS_DIR, 'metrics.json')
    save_metrics_to_json(combined_metrics, json_path)
    print(f"Saved metrics JSON: {json_path}")

    # Generate figures
    generate_target_distribution_plot(y, FIGURES_DIR)
    generate_confusion_matrix_plot(dummy_metrics, lr_metrics, FIGURES_DIR)
    generate_metrics_comparison_plot(dummy_metrics, lr_metrics, FIGURES_DIR)
    generate_roc_and_pr_curves_plot(y_test, y_prob_lr, FIGURES_DIR)
    generate_top_coefficients_plot(lr_pipe, num_cols, cat_cols, FIGURES_DIR)
    print("All 5 modeling figures generated successfully!")


if __name__ == '__main__':
    main()
