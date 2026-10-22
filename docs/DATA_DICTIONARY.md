# Data Dictionary: Diabetes Inpatient Clinical Dataset

## 1. Dataset Overview
- **Dataset Source**: UCI Machine Learning Repository (*Diabetes 130-US Hospitals 1999–2008*)
- **Total Rows**: 101,763 cleaned hospital encounters
- **Total Columns**: 54 columns in `data/processed/diabetic_data_clean.csv`
- **Unique Patients**: 71,515 individuals across 130 clinical facilities

---

## 2. Identifiers & Demographics

| Column Name | Data Type | Missing Treatment | Description & Value Domain |
| :--- | :--- | :--- | :--- |
| `encounter_id` | Integer | None (0 nulls) | Unique identifier of each hospital inpatient encounter. Dropped before modeling. |
| `patient_nbr` | Integer | None (0 nulls) | Unique identifier for each patient. Used as the grouping variable for `GroupShuffleSplit` (0 patient overlap). Dropped before modeling. |
| `race` | Categorical | Imputed | Patient race/ethnicity: `Caucasian`, `AfricanAmerican`, `Hispanic`, `Asian`, `Other`, `Unknown`. Sentinel `'?'` converted to `'Unknown'`. |
| `gender` | Categorical | Filtered | Patient biological sex: `Female` (53.8%), `Male` (46.2%). Encounters with `'Unknown/Invalid'` (3 rows) pruned in Task 1. |
| `age` | Categorical | Validated | Age bracket in 10-year intervals: `[0-10)`, `[10-20)`, ..., `[90-100)`. |
| `age_midpoint` | Float | Derived | Numeric midpoint of `age` bracket (e.g. `[50-60)` $\to$ `55.0`). Used as a continuous model input. |

---

## 3. Administrative & Admission Details

| Column Name | Data Type | Missing Treatment | Description & Value Domain |
| :--- | :--- | :--- | :--- |
| `admission_type_id` | Integer | Mapped | Raw integer code for admission urgency. Mapped to `admission_type_desc`. Dropped before modeling. |
| `admission_type_desc` | Categorical | Standardized | Admission urgency: `Emergency`, `Urgent`, `Elective`, `Not Available / Unmapped`. |
| `admission_source_id` | Integer | Mapped | Raw integer code for referral source. Mapped to `admission_source_desc`. Dropped before modeling. |
| `admission_source_desc` | Categorical | Standardized | Referral source: `Physician Referral`, `Emergency Room`, `Clinic Referral`, `Transfer from a hospital`, etc. |
| `discharge_disposition_id` | Integer | Mapped | **LEAKAGE EXCLUSION**. Raw discharge code. Excluded from predictive models to prevent target leakage. |
| `discharge_disposition_desc` | Categorical | Standardized | **LEAKAGE EXCLUSION**. Text description of discharge destination. Contains terminal events (`Expired`). Excluded from predictive models. |
| `payer_code` | Categorical | Retained | Payer/insurance category: `MC` (Medicare), `MD` (Medicaid), `BC` (Blue Cross), `SP` (Self-Pay), `Unknown`, etc. Implicit sentinel `'?'` mapped to `'Unknown'`. |
| `medical_specialty` | Categorical | Retained | Admitting clinical specialty: `InternalMedicine`, `Emergency/Trauma`, `Cardiology`, `Surgery-General`, `Missing`, etc. Implicit `'?'` mapped to `'Missing'`. |

---

## 4. Inpatient Clinical Measurements & Utilization History

| Column Name | Data Type | Missing Treatment | Description & Value Domain |
| :--- | :--- | :--- | :--- |
| `time_in_hospital` | Integer | None (0 nulls) | Inpatient length of stay in days (integer range: 1 to 14). |
| `num_lab_procedures` | Integer | None (0 nulls) | Count of distinct diagnostic laboratory procedures performed during the encounter (1 to 132). |
| `num_procedures` | Integer | None (0 nulls) | Count of non-lab medical/surgical procedures performed during the encounter (0 to 6). |
| `num_medications` | Integer | None (0 nulls) | Count of distinct pharmaceutical medications administered during the stay (1 to 81). |
| `number_outpatient` | Integer | None (0 nulls) | Count of outpatient medical visits in the 12 months preceding admission (0 to 42). |
| `number_emergency` | Integer | None (0 nulls) | Count of emergency room visits in the 12 months preceding admission (0 to 76). |
| `number_inpatient` | Integer | None (0 nulls) | Count of acute inpatient admissions in the 12 months preceding admission (0 to 21). Strongest single predictor in model ($\beta = +0.3517$). |
| `number_diagnoses` | Integer | None (0 nulls) | Total number of ICD-9 diagnosis codes entered on the hospital bill (1 to 16). |

---

## 5. Diagnostic Codes & Disease Categories

| Column Name | Data Type | Missing Treatment | Description & Value Domain |
| :--- | :--- | :--- | :--- |
| `diag_1`, `diag_2`, `diag_3` | Object | Standardized | Raw alphanumeric ICD-9 diagnostic codes. High-cardinality strings (900+ distinct codes). Dropped after mapping. |
| `diag_1_category` | Categorical | Derived | Primary ICD-9 disease category: `Circulatory` (390–459, 785), `Respiratory` (460–519, 786), `Digestive` (520–579, 787), `Diabetes` (250.xx), `Injury` (800–999), `Musculoskeletal` (710–739), `Genitourinary` (580–629, 788), `Neoplasms` (140–239), `Other`. |
| `diag_2_category` | Categorical | Derived | Secondary ICD-9 disease category mapped using identical clinical boundaries. |
| `diag_3_category` | Categorical | Derived | Tertiary ICD-9 disease category mapped using identical clinical boundaries. |

---

## 6. Diabetic Laboratory Markers

| Column Name | Data Type | Missing Treatment | Description & Value Domain |
| :--- | :--- | :--- | :--- |
| `max_glu_serum` | Categorical | Standardized | Blood glucose serum measurement: `>300`, `>200`, `Norm`, `Not Tested`. Sentinel `'None'` and `NaN` mapped to `'Not Tested'`. |
| `A1Cresult` | Categorical | Standardized | Glycated hemoglobin test result: `>8`, `>7`, `Norm`, `Not Tested`. Sentinel `'None'` and `NaN` mapped to `'Not Tested'`. |

---

## 7. Diabetic Medications & Titration (23 Features)

| Column Names | Data Type | Values | Description |
| :--- | :--- | :--- | :--- |
| `metformin`, `repaglinide`, `nateglinide`, `chlorpropamide`, `glimepiride`, `acetohexamide`, `glipizide`, `glyburide`, `tolbutamide`, `pioglitazone`, `rosiglitazone`, `acarbose`, `miglitol`, `troglitazone`, `tolazamide`, `insulin`, `glyburide-metformin`, `glipizide-metformin`, `glimepiride-pioglitazone`, `metformin-rosiglitazone`, `metformin-pioglitazone` | Categorical | `No`, `Steady`, `Up`, `Down` | Tracks dosage changes or maintenance for 21 specific antidiabetic compounds during the inpatient encounter. |
| `change` | Categorical | `Ch`, `No` | Indicates whether any diabetes medication dosage was changed or adjusted during the hospital stay. `Ch` strongly correlates with elevated readmission ($\chi^2 = 215.94$). |
| `diabetesMed` | Categorical | `Yes`, `No` | Indicates whether any diabetic medication was prescribed or administered during the encounter. |

---

## 8. Target Outcome Variable

| Column Name | Data Type | Value Domain | Description |
| :--- | :--- | :--- | :--- |
| `readmitted` | Categorical | `<30`, `>30`, `NO` | Original 3-class hospital readmission outcome. `<30` (11,357 encounters, 11.16%), `>30` (35,545 encounters, 34.93%), `NO` (54,861 encounters, 53.91%). |
| Binary Mapping | Integer | `1` (Positive), `0` (Negative) | `1` if `readmitted == '<30'` (early readmission within 30 days); `0` otherwise. |
