# Model Card: Diabetic Inpatient 30-Day Readmission Risk Estimator

## 1. Model Overview
- **Model Name**: Balanced Logistic Regression for Inpatient 30-Day Readmission
- **Version**: 1.0.0 (Educational & Retrospective Research Prototype)
- **Model Type**: Supervised Binary Classification (`scikit-learn.pipeline.Pipeline`)
- **Pipeline Components**:
  - `prep`: `ColumnTransformer` applying `StandardScaler` to 9 numerical features and `OneHotEncoder(handle_unknown='ignore')` to 34 categorical features.
  - `clf`: `LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)`
- **License**: MIT License
- **Framework**: Scikit-Learn 1.6+ / Python 3.12

---

## 2. Intended Use & Out-of-Scope Use

### Intended Use
- **Primary Use Case**: Retrospective educational benchmark and exploratory analytics demonstrating imbalanced classification handling, anti-leakage governance, and patient-level grouped evaluation in healthcare tabular data.
- **Intended Audience**: Data scientists, machine learning engineers, and clinical informatics researchers reviewing methodology and operational trade-offs in imbalanced healthcare analytics.

### Out-of-Scope & Prohibited Uses
- **Bedside Diagnostic / Triage Tool**: This model is **not a certified medical device** (not FDA/CE marked). It must **never** be used for clinical decision-making, patient discharge triage, treatment withholding, or altering physician prescription orders.
- **Modern Population Generalization**: The model must not be assumed to generalize directly to modern inpatient populations without extensive external validation and retraining, as pharmaceutical regimens and EHR recording practices have significantly changed since the study period.
- **Causal Interpretation**: Standardized regression coefficients reflect observational statistical associations within this dataset and must not be interpreted as causal clinical risk factors.

---

## 3. Training and Evaluation Data

### Dataset Provenance
- **Dataset**: Diabetes 130-US Hospitals (1999–2008), sourced from the **UCI Machine Learning Repository** ([Link](https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008)).
- **Clinical Setting**: Inpatient admissions across 130 participating US clinical medical centers.
- **Total Cleaned Volume**: 101,763 hospital encounters across 71,515 unique patients.
- **Unit of Observation**: A single hospital inpatient stay lasting between 1 and 14 days.

### Target Variable Formulation
- **Target Name**: `readmitted_30d` (binary: 0 or 1).
- **Positive Class (`1`)**: Hospital encounter with recorded readmission within 30 days of discharge (`readmitted == '<30'`).
  - Overall Dataset Prevalence: 11,357 encounters (**11.16%**).
  - Test Cohort Prevalence: 2,195 encounters (**10.92%**).
- **Negative Class (`0`)**: Hospital encounter with readmission recorded after 30 days (`>30`) or no recorded readmission (`NO`).
  - Overall Dataset Prevalence: 90,406 encounters (**88.84%**).
  - Test Cohort Prevalence: 17,898 encounters (**89.08%**).

---

## 4. Anti-Leakage Governance & Split Protocol

### Patient-Grouped Holdout Split
- **Method**: `GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=42)` grouped strictly on `patient_nbr`.
- **Partition Counts**:
  - **Training Cohort**: 81,670 encounters (57,212 unique patients).
  - **Held-Out Test Cohort**: 20,093 encounters (14,303 unique patients).
- **Patient Overlap**: Exactly **0 patient overlap** between training and test partitions.
- **Rationale**: Many diabetic patients experience recurrent hospitalizations. A naive random split would place encounters from the same patient across train and test sets, causing optimistic data leakage and overfitting to individual patient biological baselines.

### Target Leakage Feature Pruning
- **Excluded Features**: `discharge_disposition_id` and `discharge_disposition_desc`.
- **Reason**: Certain discharge disposition codes capture post-discharge terminal outcomes such as inpatient mortality (`Expired`). In the dataset, 1,642 patients with `Expired` status have an exact 0.00% readmission rate. Retaining terminal discharge codes produces artificial predictive leakage that does not reflect real-time pre-discharge care decisions.
- **Identifiers**: `encounter_id` and `patient_nbr` are dropped before feature matrix construction.

---

## 5. Quantitative Performance (Held-Out Test Cohort: N = 20,093)

All metrics below are computed on the held-out test cohort ($N = 20,093$ encounters, 2,195 positive readmission cases).

| Metric | Dummy Classifier (Majority Class) | Balanced Logistic Regression | Operational Interpretation |
| :--- | :---: | :---: | :--- |
| **Accuracy** | **89.08%** | **65.86%** | Baseline achieves misleadingly high accuracy by guessing 0 for all cases. |
| **Precision** | 0.00% | **16.48%** | ~1 in 6 encounters flagged by the model represents a true 30-day readmission. |
| **Recall (Sensitivity)** | 0.00% | **52.26%** | Identified 1,147 positive cases in the held-out test set at the selected threshold (vs 0 for baseline). |
| **F1-Score** | 0.0000 | **0.2506** | Harmonic mean of precision and recall under class imbalance. |
| **ROC-AUC** | 0.5000 | **0.6435** | Moderate discriminative power above random chance (0.5000). |
| **PR-AUC (Avg Precision)** | 0.1092 | **0.1887** | **+72.8% relative gain** over baseline test prevalence (0.1092). |

### Test Confusion Matrix Breakdown (Threshold = 0.50)
- **True Negatives (TN)**: 12,086
- **False Positives (FP)**: 5,812
- **False Negatives (FN)**: 1,048
- **True Positives (TP)**: 1,147

---

## 6. Calibration & Risk Tier Definitions

- **Probability Calibration Status**: **Uncalibrated**.
- **Model Output**: The predicted probability from `model.predict_proba()` is an uncalibrated logistic function output under `class_weight='balanced'`. Because class weights shift the intercept to balance effective class frequencies, raw probability values serve as a relative ranking score rather than an absolute frequentist probability.
- **Analytical Risk Tiers**:
  - **Elevated Risk ($\ge 50\%$)**: Meets standard decision threshold; captures 52.3% of true readmission cases in the test set.
  - **Intermediate Risk ($30\% - 49\%$)**: Exceeds test base rate prevalence (10.92%); indicates moderate chronic comorbidity or historical utilization.
  - **Baseline Risk ($< 30\%$)**: Corresponds to lower historical utilization and uncomplicated admission parameters.
- *Notice: Risk tiers are heuristic groupings defined for analytical demonstration and are not clinically validated triage cutoffs.*

---

## 7. Model Risk Drivers (Standardized Log-Odds Coefficients)

Top positive and negative standardized coefficients from multivariable regression on the training set:

1. **`number_inpatient` ($\beta = +0.3517, p < 10^{-15}$)**: Strongest positive driver. Readmission risk rises from 8.44% (0 prior stays) to 34.58% (6+ stays).
2. **`number_diagnoses` ($\beta = +0.0933$)**: Captures cumulative multi-morbidity burden.
3. **`age_midpoint` ($\beta = +0.0777$)**: Geriatric age groups exhibit elevated readmission vulnerability.
4. **Specialty Flags**: Inpatient encounters managed under `Allergy and Immunology` (+1.03) and `Hematology` (+0.98) show elevated relative coefficients, reflecting complex oncological and immunological conditions.
5. **Protective / Negative Drivers**: Certain elective surgery types and pediatric age categories display negative standardized log-odds.

---

## 8. Ethical, Fairness & Safety Considerations

- **Demographic Bias**: The dataset comprises historical admissions predominantly categorized as Caucasian (75.0%) and African American (19.0%), with lower representation of Asian, Hispanic, and other groups. Performance across subpopulations should be carefully audited before adapting methods to diverse populations.
- **Historical Pharmaceuticals**: Therapies developed after 2008 (such as SGLT2 inhibitors and GLP-1 receptor agonists) are absent from this historical cohort.
- **Clinical Safety Disclaimer**: Retrospective machine learning models cannot replace clinical assessment, diagnostic testing, or clinical judgment by qualified medical practitioners.
