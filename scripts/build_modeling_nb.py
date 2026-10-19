import os
import nbformat as nbf

nb_path = os.path.join('notebooks', '03_modeling_and_evaluation.ipynb')
with open(nb_path, 'r', encoding='utf-8') as f:
    nb = nbf.read(f, as_version=4)

m5 = """---
## Section 5: Establish a Baseline (DummyClassifier)

Before training any complex model, a good data scientist always establishes a naive baseline. Here we use scikit-learn's **`DummyClassifier`** configured with `strategy='most_frequent'`, which always predicts the dominant class (0 — Not Readmitted within 30 Days).
"""

c5 = """from sklearn.dummy import DummyClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

# Fit majority-class dummy baseline on training data
dummy = DummyClassifier(strategy='most_frequent')
dummy.fit(X_train, y_train)

# Evaluate on held-out test data
y_pred_dummy = dummy.predict(X_test)

dummy_acc = accuracy_score(y_test, y_pred_dummy)
dummy_prec = precision_score(y_test, y_pred_dummy, zero_division=0)
dummy_rec = recall_score(y_test, y_pred_dummy, zero_division=0)
dummy_f1 = f1_score(y_test, y_pred_dummy, zero_division=0)

print("=== BASELINE DUMMY CLASSIFIER (MOST FREQUENT) METRICS ===")
print(f"Accuracy:  {dummy_acc:.4f} ({dummy_acc*100:.2f}%)")
print(f"Precision: {dummy_prec:.4f}")
print(f"Recall:    {dummy_rec:.4f} (0.0% of positive cases detected)")
print(f"F1-Score:  {dummy_f1:.4f}")
print("\\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred_dummy))
"""

m5_interp = """### Why Accuracy is Deceptive on Imbalanced Datasets
Notice that the DummyClassifier achieves an accuracy of **89.08%**! To an untrained observer, an 89% accuracy might look like a working model.

However, examining the confusion matrix reveals that the model predicted `0` for all 20,093 test encounters:
- **True Negatives**: 17,898 (all negative cases)
- **True Positives**: **0**
- **False Negatives**: **2,195** (missed 100% of the actual 30-day readmissions)

In a hospital setting, this model is completely useless. It fails to identify a single high-risk patient who needs post-discharge transitional care. This illustrates why **Recall**, **Precision**, and **PR-AUC** are the essential evaluation metrics for imbalanced healthcare datasets.
"""

m6 = """---
## Section 6: Train a Simple Machine Learning Model (Logistic Regression)

### Algorithm Choice
We select **Logistic Regression** as our primary machine learning model. Logistic Regression is:
- Fast and computationally efficient.
- Interpretable, allowing inspection of standardized feature log-odds coefficients.
- Transparent for pair programming and peer review.

### Preprocessing and Class Imbalance Handling
1. **Numerical Features**: Scaled using `StandardScaler` (zero mean, unit variance) so features with larger ranges (like `num_lab_procedures` up to 132) do not dominate features with small counts (like `number_inpatient`).
2. **Categorical Features**: Encoded using `OneHotEncoder(handle_unknown='ignore')` to handle unseen categories at test time gracefully.
3. **Class Weighting**: We set `class_weight='balanced'`. This automatically calculates class weights inversely proportional to class frequencies:
   $$\\text{weight}(c) = \\frac{N}{2 \\times N_c}$$
   For our positive class, this assigns approximately **~8x higher penalty** to False Negatives, forcing the model to pay attention to readmitted patients.
"""

c6 = """from sklearn.linear_model import LogisticRegression

# Construct the full scikit-learn pipeline
lr_pipeline = build_logistic_regression_pipeline(
    numerical_cols=num_cols,
    categorical_cols=cat_cols,
    class_weight='balanced',
    max_iter=1000,
    random_state=42
)

print("Pipeline Architecture:")
print(lr_pipeline)

# Fit strictly on the training partition
print("\\nFitting Balanced Logistic Regression on X_train (81,670 encounters)...")
lr_pipeline.fit(X_train, y_train)
print("Model training complete.")
"""

nb.cells.extend([
    nbf.v4.new_markdown_cell(m5),
    nbf.v4.new_code_cell(c5),
    nbf.v4.new_markdown_cell(m5_interp),
    nbf.v4.new_markdown_cell(m6),
    nbf.v4.new_code_cell(c6)
])

with open(nb_path, 'w', encoding='utf-8') as f:
    nbf.write(nb, f)

print(f"Updated {nb_path} with Sections 5 and 6.")
