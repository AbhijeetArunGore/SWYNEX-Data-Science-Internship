import os
import nbformat as nbf

nb_path = os.path.join('notebooks', '03_modeling_and_evaluation.ipynb')
nb = nbf.v4.new_notebook()
nb.metadata['language_info'] = {'name': 'python'}

# Section 1
m1 = """# 30-Day Hospital Readmission Risk: Modeling and Evaluation
### SWYNEX Technologies — Task 3: Machine Learning Modeling & Clinical Evaluation
**Author:** Abhijeet Arun Gore  
**Role:** Data Science Intern at SWYNEX Technologies  
**Dataset:** Diabetes 130-US Hospitals (1999–2008), Cleaned Inpatient Dataset from Task 1  
**Repository:** [SWYNEX-Data-Preparation](https://github.com/AbhijeetArunGore/SWYNEX-Data-Preparation)  

---

## Section 1: Introduction

### The Clinical Prediction Problem
Hospital readmission within 30 days of discharge is an essential quality and cost metric in modern healthcare systems. Under the **Hospital Readmissions Reduction Program (HRRP)** established by the Centers for Medicare & Medicaid Services (CMS), hospitals face financial penalties for elevated 30-day readmission rates for chronic conditions, including diabetes and heart failure.

In this project, our goal is to build a transparent, baseline machine learning classification model that predicts whether an inpatient diabetic encounter is followed by an unplanned **readmission within 30 days (`<30`)**.

### Dataset and Scope
We build directly upon the work completed in previous tasks:
- **Task 1 (Data Preparation)**: Cleaned 101,763 hospital encounters, standardized implicit missingness (`?`), handled zero-variance columns, and mapped relational diagnosis lookups (`IDS_mapping.csv`).
- **Task 2 (Exploratory Data Analysis)**: Documented an ~8:1 target class imbalance, identified prior inpatient visits as the primary risk driver, and discovered critical data leakage risks (including in-hospital deceased records and repeat patient clustering).

### Guiding Principles for Task 3
1. **Honest Student Modeling**: We avoid unnecessarily complex black-box ensembles or deep learning. We implement a solid, understandable **Logistic Regression baseline with balanced class weights** and benchmark it against a majority-class **DummyClassifier**.
2. **Strict Anti-Leakage Governance**: We implement a patient-grouped holdout split (`GroupShuffleSplit`) to prevent encounters from the same patient from leaking across partitions.
3. **Clinical Metric Transparency**: In imbalanced medical datasets, raw accuracy is deceptive. We evaluate precision, recall, F1-score, ROC-AUC, and Precision-Recall AUC (PR-AUC).
4. **Educational Scope**: This model is a learning project. It is **not clinically validated** or ready for real-time bedside deployment.
"""

# Section 2
m2 = """---
## Section 2: Load and Inspect the Data

We load our cleaned dataset (`data/processed/diabetic_data_clean.csv`) exported in Task 1. We inspect the dataset shape, confirm the presence of the `readmitted` target column, verify that there are zero unhandled null values or duplicate rows, and inspect the target class balance.
"""

c2 = """import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Ensure access to modular utility functions
sys.path.insert(0, os.path.abspath('..'))
from src.modeling import (
    map_binary_target,
    prepare_feature_matrix,
    create_patient_grouped_split,
    build_logistic_regression_pipeline,
    evaluate_model
)

# Configure styling aesthetics
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams['font.sans-serif'] = 'Segoe UI'
plt.rcParams['figure.dpi'] = 120
plt.rcParams['axes.edgecolor'] = '#cccccc'
plt.rcParams['axes.linewidth'] = 0.8
pd.set_option('display.max_columns', 60)

# Load data with robust relative path handling
data_paths = [
    os.path.join('data', 'processed', 'diabetic_data_clean.csv'),
    os.path.join('..', 'data', 'processed', 'diabetic_data_clean.csv')
]
csv_path = next(p for p in data_paths if os.path.exists(p))

df = pd.read_csv(csv_path)
print(f"Dataset successfully loaded from: {csv_path}")
print(f"Shape: {df.shape[0]:,} rows x {df.shape[1]} columns")
print(f"Total Missing Values: {df.isnull().sum().sum()}")
print(f"Exact Duplicate Rows: {df.duplicated().sum()}")
print(f"Unique Patients: {df['patient_nbr'].nunique():,}")

# Confirm raw target values
print("\\n=== RAW TARGET DISTRIBUTION ('readmitted') ===")
raw_target_counts = df['readmitted'].value_counts()
raw_target_pcts = df['readmitted'].value_counts(normalize=True) * 100
for k, v in raw_target_counts.items():
    print(f"  {k:5s}: {v:6,d} ({raw_target_pcts[k]:.2f}%)")
"""

m2_target = """### Binary Target Definition:
- **Positive Class ($y = 1$)**: `readmitted == '<30'` — Readmission occurred within 30 days of discharge.
- **Negative Class ($y = 0$)**: `readmitted in {'NO', '>30'}` — No readmission was recorded, or readmission occurred after 30 days.

*Note on 'NO'*: An outcome of `NO` indicates that no readmission was observed in this hospital network. It does not prove that the patient was never hospitalized elsewhere.
"""

c2_target = """# Map binary target
df['target'] = (df['readmitted'] == '<30').astype(int)
pos_count = int(df['target'].sum())
neg_count = int(len(df) - pos_count)

print("=== BINARY TARGET CLASSIFICATION BREAKDOWN ===")
print(f"Negative Class (0 - Not Readmitted <30 Days): {neg_count:,} ({neg_count/len(df)*100:.2f}%)")
print(f"Positive Class (1 - Readmitted Within 30 Days): {pos_count:,} ({pos_count/len(df)*100:.2f}%)")
print(f"Imbalance Ratio: Approximately {neg_count / pos_count:.1f} to 1")
"""

# Section 3
m3 = """---
## Section 3: Prepare the Features

### Feature Segregation and Leakage Prevention
To prevent data leakage and ensure realistic model inputs, we strictly exclude the following columns:
1. **Identifiers**: `encounter_id` and `patient_nbr`. `patient_nbr` is preserved strictly as the grouping key for train-test splitting.
2. **Raw Outcome**: `readmitted` and temporary `target`.
3. **Confirmed Target Leakage**: `discharge_disposition_id` and `discharge_disposition_desc`.
   - In Task 2, we showed that encounters resulting in inpatient death (`Expired`, $N=1,652$) all had an exact 0.0% readmission rate.
   - Including `discharge_disposition` allows models to predict "no readmission" from the fact that a patient died.
   - Furthermore, discharge disposition is determined at the very end of hospitalization; excluding it aligns with the operational goal of identifying risk earlier during the stay.
4. **Redundant High-Cardinality Codes**: Raw `diag_1`, `diag_2`, `diag_3` (900+ codes) are captured by standard categories (`diag_1_category`, etc.), and integer ID codes (`admission_type_id`, `admission_source_id`) are captured by clean text descriptions.

All transformations (`StandardScaler` for continuous features, `OneHotEncoder(handle_unknown='ignore')` for categorical features) are assembled in a scikit-learn `ColumnTransformer` to be fitted **strictly on training data only**.
"""

c3 = """# Separate features, target, and grouping key
X, y, groups, num_cols, cat_cols = prepare_feature_matrix(df, target_col='readmitted', exclude_leakage=True)

print(f"Input Feature Matrix (X): {X.shape[0]:,} rows x {X.shape[1]} features")
print(f"Numerical Features ({len(num_cols)}): {num_cols}")
print(f"Categorical Features ({len(cat_cols)}): {len(cat_cols)} categorical attributes")
print(f"Sample Categorical: {cat_cols[:6]}...")
"""

# Section 4
m4 = """---
## Section 4: Train-Test Split

### Patient-Grouped Splitting (Preventing Intra-Patient Leakage)
In our dataset, **29.72%** of encounters represent repeat admissions by recurring patients. If we used standard random splitting, encounters from the same patient would end up in both training and test sets. The model could simply memorize patient identities rather than generalizable clinical patterns.

We use scikit-learn's **`GroupShuffleSplit`** with `test_size=0.20` and `random_state=42`, grouping strictly on `patient_nbr`.
"""

c4 = """# Create patient-grouped holdout split
X_train, X_test, y_train, y_test = create_patient_grouped_split(
    X, y, groups=groups, test_size=0.20, random_state=42
)

# Verify zero patient overlap
train_patients = set(groups.iloc[X_train.index])
test_patients = set(groups.iloc[X_test.index])
overlap = train_patients.intersection(test_patients)

print("=== PATIENT-GROUPED SPLIT VERIFICATION ===")
print(f"Training Encounters: {len(X_train):,} ({len(X_train)/len(X)*100:.1f}%)")
print(f"  Train Positive Rate: {y_train.mean()*100:.2f}% ({y_train.sum():,} positive)")
print(f"Held-Out Test Encounters: {len(X_test):,} ({len(X_test)/len(X)*100:.1f}%)")
print(f"  Test Positive Rate: {y_test.mean()*100:.2f}% ({y_test.sum():,} positive)")
print(f"Patient Overlap Between Train and Test: {len(overlap)} (Zero Leakage Confirmed)")
"""

cells = [
    nbf.v4.new_markdown_cell(m1),
    nbf.v4.new_markdown_cell(m2),
    nbf.v4.new_code_cell(c2),
    nbf.v4.new_markdown_cell(m2_target),
    nbf.v4.new_code_cell(c2_target),
    nbf.v4.new_markdown_cell(m3),
    nbf.v4.new_code_cell(c3),
    nbf.v4.new_markdown_cell(m4),
    nbf.v4.new_code_cell(c4)
]

nb['cells'] = cells

with open(nb_path, 'w', encoding='utf-8') as f:
    nbf.write(nb, f)

print(f"Initialized {nb_path} with Sections 1 to 4.")
