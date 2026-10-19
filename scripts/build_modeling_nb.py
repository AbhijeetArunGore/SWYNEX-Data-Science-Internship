import os
import nbformat as nbf

nb_path = os.path.join('notebooks', '03_modeling_and_evaluation.ipynb')
with open(nb_path, 'r', encoding='utf-8') as f:
    nb = nbf.read(f, as_version=4)

m7 = """---
## Section 7: Evaluate the Model

We evaluate both the Dummy Baseline and the Logistic Regression model on the exact same **held-out test set of 20,093 encounters**.

We calculate:
- **Accuracy**: Overall fraction of correct predictions.
- **Precision**: When the model flags an encounter as high-risk, how often is it right?
- **Recall (Sensitivity)**: What percentage of actual 30-day readmissions did the model capture?
- **F1-Score**: Harmonic mean of Precision and Recall.
- **ROC-AUC**: Area under the Receiver Operating Characteristic curve.
- **PR-AUC (Average Precision)**: Area under the Precision-Recall curve, specifically calibrated for imbalanced data.
"""

c7 = """from sklearn.metrics import classification_report, roc_auc_score, average_precision_score

# Generate predictions and probabilities for test partition
y_pred_lr = lr_pipeline.predict(X_test)
y_prob_lr = lr_pipeline.predict_proba(X_test)[:, 1]
y_prob_dummy = dummy.predict_proba(X_test)[:, 1]

# Evaluate both models using modular utility
dummy_metrics = evaluate_model(dummy, X_test, y_test)
lr_metrics = evaluate_model(lr_pipeline, X_test, y_test)

metrics_comparison = pd.DataFrame([
    {
        'Evaluation Metric': 'Accuracy',
        'Dummy Baseline': f"{dummy_metrics['accuracy']:.4f}",
        'Balanced Logistic Regression': f"{lr_metrics['accuracy']:.4f}",
        'Clinical Significance': 'Baseline has high accuracy by guessing 0; model trades overall accuracy for sensitivity.'
    },
    {
        'Evaluation Metric': 'Precision',
        'Dummy Baseline': f"{dummy_metrics['precision']:.4f}",
        'Balanced Logistic Regression': f"{lr_metrics['precision']:.4f}",
        'Clinical Significance': '16.48% of flagged patients readmit within 30 days (~1 in 6 alarms is true).'
    },
    {
        'Evaluation Metric': 'Recall (Sensitivity)',
        'Dummy Baseline': f"{dummy_metrics['recall']:.4f}",
        'Balanced Logistic Regression': f"{lr_metrics['recall']:.4f}",
        'Clinical Significance': 'Model catches 1,147 of 2,195 (52.26%) true 30-day readmissions.'
    },
    {
        'Evaluation Metric': 'F1-Score',
        'Dummy Baseline': f"{dummy_metrics['f1_score']:.4f}",
        'Balanced Logistic Regression': f"{lr_metrics['f1_score']:.4f}",
        'Clinical Significance': 'Harmonic balance between precision and recall on the positive class.'
    },
    {
        'Evaluation Metric': 'ROC-AUC',
        'Dummy Baseline': f"{dummy_metrics['roc_auc']:.4f}",
        'Balanced Logistic Regression': f"{lr_metrics['roc_auc']:.4f}",
        'Clinical Significance': 'Overall discriminatory capability across thresholds (0.50 = random chance).'
    },
    {
        'Evaluation Metric': 'PR-AUC (Average Precision)',
        'Dummy Baseline': f"{dummy_metrics['pr_auc']:.4f}",
        'Balanced Logistic Regression': f"{lr_metrics['pr_auc']:.4f}",
        'Clinical Significance': '+72.8% relative improvement over test positive base rate (0.1092).'
    }
])

print("=== SIDE-BY-SIDE MODEL EVALUATION ON HELD-OUT TEST COHORT (N=20,093) ===")
metrics_comparison
"""

c7_report = """print("=== DETAILED CLASSIFICATION REPORT: LOGISTIC REGRESSION ===")
print(classification_report(y_test, y_pred_lr, digits=4, zero_division=0))
"""

m8 = """---
## Section 8: Visualize the Results

We create four key visualizations to communicate model behavior:
1. **Confusion Matrix Comparison**: Visualizes True Positives, False Positives, False Negatives, and True Negatives.
2. **Metrics Bar Chart**: Side-by-side comparison across all 6 performance metrics.
3. **ROC and Precision-Recall Curves**: Illustrates trade-offs across all continuous classification thresholds.
4. **Top Feature Coefficients**: Displays the strongest positive and negative predictors of 30-day readmission.
"""

c8_cm = """# 1. Confusion Matrix Comparison
fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))

sns.heatmap(np.array(dummy_metrics['confusion_matrix']), annot=True, fmt=',d', cmap='Blues', cbar=False,
            xticklabels=['Pred 0 (No)', 'Pred 1 (<30)'], yticklabels=['True 0 (No)', 'True 1 (<30)'],
            ax=axes[0], annot_kws={'size': 11, 'weight': 'bold'})
axes[0].set_title('A: Baseline DummyClassifier\n(Recall = 0.0%, Misses All 2,195 Cases)', fontsize=11, fontweight='bold')
axes[0].set_xlabel('Predicted Label', fontsize=10)
axes[0].set_ylabel('True Label', fontsize=10)

sns.heatmap(np.array(lr_metrics['confusion_matrix']), annot=True, fmt=',d', cmap='Blues', cbar=False,
            xticklabels=['Pred 0 (No)', 'Pred 1 (<30)'], yticklabels=['True 0 (No)', 'True 1 (<30)'],
            ax=axes[1], annot_kws={'size': 11, 'weight': 'bold'})
axes[1].set_title(f"B: Balanced Logistic Regression\\n(Recall = 52.26%, Catches 1,147 Cases)", fontsize=11, fontweight='bold')
axes[1].set_xlabel('Predicted Label', fontsize=10)
axes[1].set_ylabel('True Label', fontsize=10)

plt.suptitle('Held-Out Test Set Confusion Matrix Comparison (N=20,093)', fontsize=13, fontweight='bold', y=1.02)
plt.tight_layout()
plt.show()
"""

c8_bars = """# 2. Metric Comparison Bar Chart
metric_labels = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC', 'PR-AUC']
dummy_scores = [dummy_metrics['accuracy'], dummy_metrics['precision'], dummy_metrics['recall'], dummy_metrics['f1_score'], dummy_metrics['roc_auc'], dummy_metrics['pr_auc']]
lr_scores = [lr_metrics['accuracy'], lr_metrics['precision'], lr_metrics['recall'], lr_metrics['f1_score'], lr_metrics['roc_auc'], lr_metrics['pr_auc']]

x = np.arange(len(metric_labels))
width = 0.35

plt.figure(figsize=(10, 5))
plt.bar(x - width/2, dummy_scores, width, label='Dummy Baseline (Most Frequent)', color='#7f8c8d', edgecolor='black', linewidth=0.5)
plt.bar(x + width/2, lr_scores, width, label='Balanced Logistic Regression', color='#2b5c8f', edgecolor='black', linewidth=0.5)

plt.title('Performance Comparison: Baseline vs. Balanced Logistic Regression', fontsize=12, fontweight='bold', pad=12)
plt.xlabel('Metric', fontsize=11, fontweight='semibold')
plt.ylabel('Score (0.0 to 1.0)', fontsize=11, fontweight='semibold')
plt.xticks(x, metric_labels, fontsize=10, fontweight='semibold')
plt.ylim(0, 1.05)
plt.legend(frameon=True, loc='upper right')

for i, (d, l) in enumerate(zip(dummy_scores, lr_scores)):
    plt.text(i - width/2, d + 0.02, f'{d:.3f}', ha='center', va='bottom', fontsize=8.5, color='#555555')
    plt.text(i + width/2, l + 0.02, f'{l:.3f}', ha='center', va='bottom', fontsize=8.5, fontweight='bold', color='#1a3a5a')

plt.tight_layout()
plt.show()
"""

c8_curves = """# 3. ROC and Precision-Recall Curves
from sklearn.metrics import roc_curve, precision_recall_curve

fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))

# ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_prob_lr)
axes[0].plot(fpr, tpr, color='#2b5c8f', linewidth=2.5, label=f'Logistic Regression (AUC = {lr_metrics["roc_auc"]:.4f})')
axes[0].plot([0, 1], [0, 1], color='#7f8c8d', linestyle='--', linewidth=1.5, label='Random Chance (AUC = 0.5000)')
axes[0].set_title('A: Receiver Operating Characteristic (ROC) Curve', fontsize=11, fontweight='bold')
axes[0].set_xlabel('False Positive Rate', fontsize=10)
axes[0].set_ylabel('True Positive Rate (Recall)', fontsize=10)
axes[0].legend(loc='lower right', frameon=True)

# PR Curve
prec_vals, rec_vals, _ = precision_recall_curve(y_test, y_prob_lr)
base_rate = float(y_test.mean())
axes[1].plot(rec_vals, prec_vals, color='#c0392b', linewidth=2.5, label=f'Logistic Regression (PR-AUC = {lr_metrics["pr_auc"]:.4f})')
axes[1].axhline(y=base_rate, color='#7f8c8d', linestyle='--', linewidth=1.5, label=f'Test Positive Base Rate ({base_rate:.4f})')
axes[1].set_title('B: Precision-Recall (PR) Curve', fontsize=11, fontweight='bold')
axes[1].set_xlabel('Recall (Sensitivity)', fontsize=10)
axes[1].set_ylabel('Precision (Positive Predictive Value)', fontsize=10)
axes[1].legend(loc='upper right', frameon=True)

plt.suptitle('Discrimination Curves on Held-Out Test Cohort', fontsize=13, fontweight='bold', y=1.02)
plt.tight_layout()
plt.show()
"""

c8_coefs = """# 4. Top Feature Coefficients from Logistic Regression
cat_encoder = lr_pipeline.named_steps['prep'].named_transformers_['cat']
cat_feature_names = cat_encoder.get_feature_names_out(cat_cols)
all_feature_names = num_cols + list(cat_feature_names)

coefs = lr_pipeline.named_steps['clf'].coef_[0]
coef_df = pd.DataFrame({'feature': all_feature_names, 'coefficient': coefs})

top_pos = coef_df.sort_values(by='coefficient', ascending=False).head(8)
top_neg = coef_df.sort_values(by='coefficient', ascending=True).head(8)
plot_coef_df = pd.concat([top_neg, top_pos]).sort_values(by='coefficient', ascending=True)

plot_coef_df['display'] = plot_coef_df['feature'].apply(
    lambda x: x.replace('medical_specialty_', 'Spec: ').replace('admission_type_desc_', 'Adm: ').replace('payer_code_', 'Payer: ')
)

plt.figure(figsize=(10, 6))
bar_colors = ['#27ae60' if c < 0 else '#c0392b' for c in plot_coef_df['coefficient']]
plt.barh(plot_coef_df['display'], plot_coef_df['coefficient'], color=bar_colors, edgecolor='black', linewidth=0.5)

plt.title('Top Positive and Negative Logistic Regression Coefficients\n(Predicting 30-Day Early Hospital Readmission)', fontsize=12, fontweight='bold', pad=12)
plt.xlabel('Standardized Log-Odds Coefficient', fontsize=11, fontweight='semibold')
plt.ylabel('Clinical Feature', fontsize=11, fontweight='semibold')
plt.axvline(x=0, color='black', linestyle='-', linewidth=0.8)

for bar in plt.gca().patches:
    w = bar.get_width()
    ha = 'left' if w >= 0 else 'right'
    offset = 0.03 if w >= 0 else -0.03
    plt.text(w + offset, bar.get_y() + bar.get_height()/2.0, f'{w:.2f}', va='center', ha=ha, fontsize=8.5, fontweight='bold')

plt.tight_layout()
plt.show()
"""

nb.cells.extend([
    nbf.v4.new_markdown_cell(m7),
    nbf.v4.new_code_cell(c7),
    nbf.v4.new_code_cell(c7_report),
    nbf.v4.new_markdown_cell(m8),
    nbf.v4.new_code_cell(c8_cm),
    nbf.v4.new_code_cell(c8_bars),
    nbf.v4.new_code_cell(c8_curves),
    nbf.v4.new_code_cell(c8_coefs)
])

with open(nb_path, 'w', encoding='utf-8') as f:
    nbf.write(nb, f)

print(f"Updated {nb_path} with Sections 7 and 8.")
