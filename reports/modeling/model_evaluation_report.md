# SWYNEX Technologies — Task 3: Modeling and Evaluation
## Clinical Machine Learning Evaluation Report

**Internship Role**: Data Science Intern, SWYNEX Technologies  
**Project**: 30-Day Hospital Readmission Risk Baseline Modeling  
**Dataset**: Diabetes 130-US Hospitals (1999–2008), Cleaned Inpatient Cohort  
**Evaluation File**: [`reports/modeling/metrics.json`](metrics.json)  

---

## 1. Prediction Objective
The objective of Task 3 is to build and evaluate a transparent baseline machine learning model to predict **early hospital readmission within 30 days** (`<30`) among diabetic inpatient encounters. 

In clinical operations, 30-day unplanned readmissions represent a key quality indicator tracked by healthcare systems and regulatory bodies (such as the Centers for Medicare & Medicaid Services HRRP). Early identification allows care coordinators to prioritize transition planning, post-discharge medication reconciliation, and home health follow-ups.

> **Important Clinical Scope**: This model is an educational baseline developed as part of an internship project. It is **not clinically validated** and is not designed or approved for autonomous medical decision-making in hospital settings.

---

## 2. Dataset and Target Definition

### Baseline Cohort
- **Total Inpatient Encounters**: 101,763 hospitalizations across 71,515 distinct patients.
- **Source**: UCI Machine Learning Repository (Diabetes 130-US Hospitals dataset, 1999–2008).
- **Missing Values**: 0 unhandled nulls (handled systematically in Task 1).

### Target Variable Formulation
The original dataset records three categories in the `readmitted` column:
- `<30`: Patient was readmitted within 30 days of discharge.
- `>30`: Patient was readmitted more than 30 days after discharge.
- `NO`: No readmission was recorded in the dataset.

We map this to a binary classification target:
- **Positive Class ($y = 1$)**: `readmitted == '<30'` (Early Readmission) — **11,357 encounters (11.16%)**
- **Negative Class ($y = 0$)**: `readmitted in {'NO', '>30'}` — **90,406 encounters (88.84%)**

> **Clinical Interpretation of 'NO'**: An entry of `NO` indicates that no readmission was observed within the historical hospital network capture window. It is not conclusive proof that the patient was never readmitted (e.g. they may have sought care at an unmonitored hospital facility).

---

## 3. Train-Test Splitting & Leakage Prevention

### Patient-Grouped Holdout Split
In Task 2, we discovered that **29.72%** of encounters represent repeat visits by the same patients (up to 40 admissions for a single individual). If encounters were split randomly, multiple encounters from the same patient would end up in both the training set and the test set, creating artificial intra-patient data leakage.

To ensure realistic out-of-sample evaluation:
- We use scikit-learn's **`GroupShuffleSplit`** grouped by `patient_nbr` with `test_size=0.20` and `random_state=42`.
- **Training Cohort**: **81,670 encounters** (11.22% positive rate)
- **Test Cohort**: **20,093 encounters** (10.92% positive rate; 2,195 positive cases, 17,898 negative cases)
- **Patient Overlap**: **Exactly 0 patients overlap** between the training and test sets.

---

## 4. Feature Selection & Leakage Exclusion

Out of 54 original prepared columns, the final model utilizes **43 clinical and administrative features** (9 numerical, 34 categorical). The following columns were deliberately excluded:

| Excluded Column | Category / Reason for Exclusion |
| :--- | :--- |
| `encounter_id` | Unique encounter identifier; carries no generalizable predictive information. |
| `patient_nbr` | Patient identifier; used strictly as the grouping key for train-test splitting. |
| `readmitted` | Raw target column containing the ground-truth outcome. |
| `discharge_disposition_id` & `discharge_disposition_desc` | **Confirmed Target Leakage & Timing Issue**: In Task 2, we proved that deceased patients (`Expired`, $N=1,652$) have an exact 0.0% readmission rate. Including this column allows models to predict "No Readmission" simply from the fact that a patient died. Furthermore, final discharge destination is unavailable during the hospital stay. |
| `admission_type_id`, `admission_source_id` | Redundant integer codes; preserved via their human-readable descriptive columns. |
| `age` | Redundant text bracket; represented as numerical `age_midpoint`. |
| `diag_1`, `diag_2`, `diag_3` | High-cardinality raw ICD-9 codes (900+ levels); captured as standard clinical categories (`diag_1_category`, etc.). |

### Feature Matrix Overview
- **Numerical Features (9)**: `time_in_hospital`, `num_lab_procedures`, `num_procedures`, `num_medications`, `number_outpatient`, `number_emergency`, `number_inpatient`, `number_diagnoses`, `age_midpoint`.
- **Categorical Features (34)**: Demographics (`race`, `gender`), payer information (`payer_code`, `medical_specialty`), admission pathways (`admission_type_desc`, `admission_source_desc`), diagnosis categories (`diag_1_category`, `diag_2_category`, `diag_3_category`), glycemic lab testing (`max_glu_serum`, `A1Cresult`), treatment indicators (`change`, `diabetesMed`), and 23 individual diabetic medications.

All preprocessing transformations are encapsulated in a scikit-learn `Pipeline` and `ColumnTransformer` (`StandardScaler` for numbers, `OneHotEncoder(handle_unknown='ignore')` for categories) **fitted strictly on training data only**.

---

## 5. Evaluation Metrics and Baseline Comparison

Because positive cases represent only 10.92% of the test set (~8:1 imbalance), raw accuracy is fundamentally misleading. We compare our **Balanced Logistic Regression** model against a **DummyClassifier (Most Frequent Class)** baseline across a multi-metric battery on the exact same 20,093 held-out encounters:

| Metric | Dummy Baseline (Majority Class) | Balanced Logistic Regression | Meaning and Practical Interpretation |
| :--- | :---: | :---: | :--- |
| **Accuracy** | **89.08%** | 65.86% | Baseline appears "accurate" simply by always guessing 0, while missing every single vulnerable patient. |
| **Precision** | 0.00% | **16.48%** | Out of all patients flagged by Logistic Regression as high-risk, ~1 in 6 is readmitted within 30 days. |
| **Recall (Sensitivity)** | 0.00% | **52.26%** | **Key Improvement**: The model successfully detects **1,147 out of 2,195** early readmissions. |
| **F1-Score** | 0.0000 | **0.2506** | Harmonic balance between precision and sensitivity. |
| **ROC-AUC** | 0.5000 | **0.6435** | Measures discrimination across all probability cutoffs (0.50 = random chance). |
| **PR-AUC (Avg Precision)** | 0.1092 | **0.1887** | **+72.8% relative improvement** over the baseline positive rate (0.1092). |

---

## 6. Confusion Matrix Breakdown (Test Set: N=20,093)

```
Dummy Baseline Confusion Matrix:           Logistic Regression Confusion Matrix:
             Pred 0      Pred 1                         Pred 0      Pred 1
True 0       17,898           0            True 0       12,086       5,812  (False Positives)
True 1        2,195           0            True 1        1,048       1,147  (True Positives)
```

### Detailed Plain-English Interpretation:
1. **True Positives (1,147)**: The model correctly identified 1,147 patients who were readmitted within 30 days. In a hospital workflow, these patients would receive post-discharge outreach.
2. **False Negatives (1,048)**: The model missed 1,048 patients who were readmitted within 30 days. This represents an ongoing challenge for a simple linear model.
3. **False Positives (5,812)**: The model flagged 5,812 patients who were not readmitted within 30 days. This reflects the trade-off of using `class_weight='balanced'`: by giving higher weight to the minority class, the model casts a wider net to capture more true readmissions, accepting more false alarms.
4. **True Negatives (12,086)**: The model correctly ruled out 12,086 patients who were not readmitted within 30 days.

---

## 7. Model Feature Drivers (Logistic Regression Coefficients)

Examining the model's standardized log-odds coefficients reveals the primary drivers of readmission risk:

### Strongest Positive Risk Factors:
1. **`number_inpatient` ($\beta = +0.3517$)**: The single strongest numerical driver in the model. Encounters with higher historical inpatient visits have substantially higher readmission log-odds.
2. **`number_diagnoses` ($\beta = +0.0933$)**: Multi-morbidity and complex clinical presentations increase post-discharge vulnerability.
3. **`age_midpoint` ($\beta = +0.0777$)**: Geriatric diabetic patients face higher acute vulnerability.
4. **`number_emergency` ($\beta = +0.0727$)**: Frequent emergency room visits indicate unstable outpatient chronic disease management.
5. **Specialty Indicators**: Patients managed by `Allergy and Immunology` ($\beta = +1.0335$), `Hematology` ($\beta = +0.9825$), and `Infectious Diseases` ($\beta = +0.8512$) had higher risk scores.

### Strongest Negative Risk Factors:
1. **Specialty Indicators**: Encounters associated with `Pediatrics-Endocrinology` ($\beta = -1.8201$) and `Gynecology` ($\beta = -1.5329$) had lower readmission probabilities.
2. **Admission Source**: Encounters admitted through `Trauma Center` ($\beta = -0.8345$) had lower 30-day readmission risk, likely reflecting acute one-time traumatic events rather than decompensated chronic disease.

---

## 8. Limitations & Honest Assessment

1. **Baseline Model Simplicity**: A single Logistic Regression with linear decision boundaries cannot capture non-linear feature interactions (such as the compounding risk of prior inpatient visits combined with specific medication combinations).
2. **Historical Data (1999–2008)**: Clinical guidelines, medications, and EHR technology have changed substantially over the past 15+ years. Results should not be generalized to modern healthcare systems without retraining.
3. **Low Precision (16.48%)**: While recall is respectable (52.26%), the false positive rate means that clinical teams would need to contact ~6 patients to reach 1 patient who would actually be readmitted.
4. **Unobserved Out-of-Hospital Events**: The dataset captures only readmissions within the monitored hospital system. Encounters at unaffiliated clinics, urgent care centers, or out-of-network hospitals are unrecorded.
5. **Association vs. Causation**: Regression coefficients represent statistical associations and must not be interpreted as causal medical drivers.

---

## 9. Conclusion & Next Steps
Task 3 successfully demonstrated the end-to-end implementation of a leak-free clinical classification pipeline. We proved that:
- Standard classification accuracy is misleading on imbalanced healthcare datasets.
- A balanced Logistic Regression model demonstrates clear discriminative utility over naive guessing, identifying **52.26% of positive cases in the held-out test set at the selected threshold (1,147 out of 2,195)** and achieving a **+72.8% relative gain in PR-AUC** over baseline test prevalence.
- Patient-grouped splitting and deliberate leakage exclusion preserve evaluation integrity.

### Future Work:
- Explore non-linear tree-based models (such as LightGBM or Random Forests) to capture non-linear clinical interactions.
- Tune the decision threshold based on hospital economic and staffing capacity rather than the default 0.50 cutoff.
- Engineer domain interaction terms (e.g., inpatient-to-outpatient visit ratios, medication-to-length-of-stay ratios).
