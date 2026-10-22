# Diabetes Inpatient 30-Day Readmission Risk Prototype

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tests: 31 Passed](https://img.shields.io/badge/tests-31%20passed-success.svg)](tests/)
[![Code Style: Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

A reproducible, leak-free machine learning investigation and interactive risk estimation prototype predicting **30-day hospital readmissions** for diabetic inpatients across 130 US hospitals (1999–2008).

---

## 1. Project Overview & Problem Formulation

### The Problem
Hospital readmissions occurring within 30 days of discharge represent a major clinical and operational challenge under value-based healthcare quality programs (such as the US Hospital Readmissions Reduction Program). In diabetic inpatients, early readmission frequently reflects transitional care vulnerability, post-discharge glycemic volatility, or unaddressed multi-morbidity.

### Primary Objective
*Can historical inpatient electronic health record (EHR) features available prior to discharge identify hospital encounters associated with a 30-day readmission, and what are the operational trade-offs and methodological limitations of making that prediction under severe class imbalance?*

---

## 2. Intended Use & Clinical Limitations

> **CRITICAL DISCLAIMER**:
> This software is an **educational and retrospective research prototype**. It is **not a certified clinical decision support system, not a diagnostic medical device, and not approved by the FDA or other regulatory bodies**. It must **never** be used for patient bedside triage, discharge determinations, altering pharmacotherapy, or withholding clinical care.

### Key Methodological Limitations:
1. **Historical Observational Data**: Sourced from 1999–2008 EHR records; modern therapeutics (such as GLP-1 receptor agonists and SGLT2 inhibitors) and continuous glucose monitors (CGM) are absent.
2. **Correlation vs. Causation**: Regression coefficients represent empirical statistical associations in observational data, not direct causal biological mechanisms.
3. **Uncalibrated Model Output**: Predicted probabilities reflect raw logistic outputs under balanced class weighting, serving as relative risk rankings rather than absolute frequentist probabilities.
4. **Administrative Billing Code Proxies**: Diagnoses are derived from ICD-9 billing codes, which reflect administrative and reimbursement coding practices.

---

## 3. Dataset Provenance & Cohort Structure

- **Source**: UCI Machine Learning Repository — [Diabetes 130-US Hospitals (1999–2008)](https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008).
- **Scope**: 101,763 cleaned inpatient admissions across **71,515 unique patients** and **130 clinical facilities**.
- **Unit of Observation**: A single inpatient encounter lasting between 1 and 14 days.
- **Target Variable Formulation**:
  - **Positive Class (`1`)**: Hospital readmission within 30 days of discharge (`readmitted == '<30'`).
    - Entire Dataset Prevalence: 11,357 encounters (**11.16%**).
    - Held-Out Test Set Prevalence: 2,195 encounters (**10.92%**).
  - **Negative Class (`0`)**: Readmission recorded after 30 days (`>30`) or no recorded readmission (`NO`).
    - Entire Dataset Prevalence: 90,406 encounters (**88.84%**).
    - Held-Out Test Set Prevalence: 17,898 encounters (**89.08%**).

---

## 4. Anti-Leakage Governance & Methodology

1. **Patient-Grouped Validation Split**:
   - Implemented `GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=42)` grouped strictly by `patient_nbr`.
   - **Training Set**: 81,670 encounters across 57,212 unique patients.
   - **Held-Out Test Set**: 20,093 encounters across 14,303 unique patients.
   - **Zero Patient Overlap**: Exactly **0 patient overlap** between training and testing sets, eliminating optimistic evaluation leakage from recurring chronic patients.
2. **Pruning Post-Discharge Terminal Leakage**:
   - Discharged patients with terminal events (such as `Expired` / inpatient mortality, comprising 1,642 encounters) have an exact **0.00% readmission rate** because deceased patients cannot be readmitted.
   - All `discharge_disposition` variables were excluded from predictive feature matrices to ensure models rely only on pre-discharge observations.
3. **Pipeline Encapsulation**:
   - `ColumnTransformer` (scaling 9 continuous variables with `StandardScaler` and encoding 34 categorical variables with `OneHotEncoder(handle_unknown='ignore')`) is encapsulated inside an end-to-end `Pipeline` fitted exclusively on training data.
   - Encounters identifiers (`encounter_id`, `patient_nbr`) are dropped prior to model fitting.

---

## 5. Verified Evaluation Results on Held-Out Test Cohort (N = 20,093)

Evaluation was performed on the held-out test cohort ($N = 20,093$ encounters, 2,195 positive readmissions):

| Evaluation Metric | Dummy Baseline (Majority Class) | Balanced Logistic Regression | Operational Meaning |
| :--- | :---: | :---: | :--- |
| **Accuracy** | **89.08%** | **65.86%** | Majority baseline exhibits misleadingly high accuracy by predicting 0 for all encounters. |
| **Precision** | 0.00% | **16.48%** | ~1 in 6 encounters flagged by the balanced model represents a true 30-day readmission. |
| **Recall (Sensitivity)** | 0.00% | **52.26%** | Identified **1,147 positive cases in the held-out test set at the selected threshold** (vs 0 for baseline). |
| **F1-Score** | 0.0000 | **0.2506** | Harmonic balance between sensitivity and precision under severe class imbalance. |
| **ROC-AUC** | 0.5000 | **0.6435** | Moderate discriminative power above random chance (0.5000). |
| **PR-AUC (Avg Precision)** | 0.1092 | **0.1887** | **+72.8% relative gain** over baseline test set prevalence (0.1092). |

### Held-Out Test Confusion Matrix Breakdown (Decision Threshold = 0.50)
- **True Negatives (TN)**: 12,086 encounters
- **False Positives (FP)**: 5,812 encounters (alerts that do not readmit within 30 days)
- **False Negatives (FN)**: 1,048 encounters (missed readmissions)
- **True Positives (TP)**: 1,147 encounters (successfully identified positive cases)

---

## 6. Installation & Reproduction

### Prerequisites
- Python 3.10+ (tested on Python 3.12)
- Git

### Setup Instructions

```bash
# Clone repository
git clone https://github.com/AbhijeetArunGore/SWYNEX-Data-Science-Internship.git
cd SWYNEX-Data-Science-Internship

# Create and activate virtual environment
# Windows (PowerShell):
python -m venv venv
.\venv\Scripts\Activate.ps1

# Linux / macOS:
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Reproduce Artifacts & Figures
```bash
# Generate exploratory analysis diagnostic figures into reports/eda/figures/
python scripts/generate_eda_figures.py

# Train baseline and balanced models, export metrics.json, and generate modeling figures
python scripts/generate_modeling_artifacts.py
```

### Run Automated Test Suite
```bash
# Run complete test suite (31 unit tests)
python -m pytest tests/ -v
```

### Run Security & Hygiene Audit
```bash
# Scan working tree and commit history for hardcoded secrets, paths, and sensitive artifacts
python scripts/security_audit.py
```

### Launch Interactive Streamlit Prototype
```bash
# Start local interactive risk estimation web application
streamlit run app.py
```
Access the application locally at: `http://localhost:8501`

---

## 7. Repository Structure

```
SWYNEX-Data-Science-Internship/
├── .github/
│   └── workflows/
│       └── test.yml                         # Automated CI pipeline
├── data/
│   ├── raw/
│   │   ├── diabetic_data.csv                # Original UCI clinical dataset (101,766 x 50)
│   │   └── IDS_mapping.csv                  # Relational mapping keys
│   └── processed/
│       ├── diabetic_data_clean.csv          # Cleaned dataset (101,763 x 54, 0 nulls)
│       └── diabetic_data_clean.parquet      # Columnar format for high-performance I/O
├── docs/
│   ├── model_card.md                        # Formal machine learning model card
│   ├── data_dictionary.md                   # Complete 54-feature clinical schema
│   ├── security.md                          # HIPAA, credential & security policy
│   └── portfolio/
│       └── task4_presentation_guide.md      # Optional portfolio presentation notes
├── notebooks/
│   ├── 01_data_preparation.ipynb            # Ingestion, profiling & standardization
│   ├── 02_exploratory_data_analysis.ipynb   # Statistical hypothesis testing & EDA
│   ├── 03_modeling_and_evaluation.ipynb     # Holdout benchmarking & discrimination curves
│   └── 04_final_data_science_case_study.ipynb# Integrated capstone case study
├── reports/
│   ├── final_case_study_report.md           # End-to-end analytical case study report
│   ├── eda/                                 # EDA summary report and figures
│   └── modeling/
│       ├── metrics.json                     # Machine-readable test evaluation metrics
│       ├── model_evaluation_report.md       # Modeling benchmark report
│       └── model_pipeline.joblib            # Serialized trained scikit-learn pipeline
├── scripts/
│   ├── compute_eda_stats.py                 # Summary statistics computation
│   ├── generate_eda_figures.py              # Visualizations generator for EDA
│   ├── generate_modeling_artifacts.py       # Pipeline training & metrics exporter
│   ├── security_audit.py                    # Secret scanner & Git hygiene auditor
│   └── validate_dataset.py                  # Preprocessing integrity validator
├── src/
│   ├── __init__.py                          # Package entry point
│   ├── eda.py                               # Statistical analysis routines
│   ├── modeling.py                          # Feature matrices, pipelines & splitting
│   ├── prediction.py                        # Validation, schema checking & inference
│   └── utils.py                             # ICD-9 mapping & age midpoint converters
├── tests/
│   ├── test_eda.py                          # Statistical test verifications (8 tests)
│   ├── test_modeling.py                     # Split isolation & pipeline tests (8 tests)
│   ├── test_preprocessing.py                # Standardization & mapping tests (3 tests)
│   └── test_prediction.py                   # Inference, schema & input tests (9 tests)
├── .env.example                             # Configuration placeholders
├── .gitignore                               # Comprehensive Git exclusions
├── app.py                                   # Streamlit prototype entry point
├── README.md                                # Project documentation
└── requirements.txt                         # Project dependencies
```

---

## 8. Security & Data Governance

- **Zero Secret Exposure**: Verified by `scripts/security_audit.py` with zero hardcoded API keys, private keys, or passwords.
- **De-Identified Data**: Conforms to institutional research guidelines; contains no direct patient identifiers.
- **In-Memory Inference**: The Streamlit application processes inputs in volatile memory without persisting or transmitting patient features.
- See [`docs/security.md`](docs/security.md) for full security and data handling policies.

---

## 9. Future Engineering Directions

1. **Probability Calibration**: Implement Isotonic Regression or Platt Scaling to transform uncalibrated logistic decision scores into true calibrated clinical probabilities.
2. **Non-Linear Tree Ensembles**: Benchmark LightGBM or XGBoost under patient-grouped cross-validation to capture non-linear interactions across multi-morbidity categories.
3. **Subgroup Fairness Auditing**: Conduct disaggregated performance audits across demographic strata (e.g. race, age brackets, gender) to quantify disparate false positive rates.
4. **Time-to-Event Survival Analysis**: Reframe readmission prediction as Cox proportional hazards or survival trees to predict continuous days-to-readmission.

---

## 10. License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
