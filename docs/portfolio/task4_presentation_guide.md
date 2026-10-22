# Portfolio Reference: Presentation Guide & LinkedIn Post Draft
## Diabetic Inpatient 30-Day Readmission Machine Learning Prototype

> **Notice**: This document is an optional portfolio reference preserved for professional demonstration, screen recording, and LinkedIn post drafting. It is excluded from the technical production application.

---

### LinkedIn Post Draft

Excited to share an end-to-end healthcare machine learning project investigating 30-day early hospital readmissions for diabetic inpatients!

Analyzing 101,763 hospital encounters across 130 US hospitals (1999–2008) from the UCI Machine Learning Repository, this project addresses the challenge of predicting early readmission within 30 days of discharge under an ~8:1 negative-to-positive class imbalance.

Core Problem & The "Accuracy Trap":
In imbalanced healthcare data (11.16% overall base rate; 10.92% on held-out test data), raw accuracy is deceptive. A naive majority-class baseline that guesses "no readmission" for every encounter achieves 89.08% accuracy, yet catches 0 readmitted patients (0% recall).

Anti-Leakage Engineering:
- Patient-Grouped Validation: Applied GroupShuffleSplit on patient_nbr to partition 81,670 training encounters and 20,093 held-out test encounters with exactly 0 patient overlap.
- Leakage Pruning: Excluded post-discharge terminal variables (discharge disposition containing inpatient mortality with 0% readmissions) so the model uses only features known before discharge.

Model Evaluation on Held-Out Test Cohort (N = 20,093):
- Recall (Sensitivity): 52.26% — Identified 1,147 positive cases in the held-out test set at the selected threshold (out of 2,195 true readmissions).
- PR-AUC (Average Precision): 0.1887 — A +72.8% relative improvement over test base rate prevalence (0.1092).
- ROC-AUC: 0.6435 — Moderate discriminative capacity above random baseline (0.5000).
- Precision: 16.48% — Approximately 1 in 6 flagged encounters results in readmission.

Operational Context:
In healthcare operations, false positives incur low-cost outreach (such as nurse phone check-ins or pharmacist medication review), whereas false negatives (missed deteriorating patients) carry substantial clinical risks. Balanced class weighting aligns the model with operational sensitivity priorities.

Key Risk Signals:
- Prior Acute Inpatient Visits (beta = +0.3517, p < 1e-15): Monotonic rise in readmission rate from 8.44% (0 visits) to 34.58% (6+ visits).
- Medication Titration: Inpatient drug adjustments correlate with glycemic volatility (chi2 = 215.94, p < 1e-46).
- Multi-Morbidity: Compounding chronic conditions (beta = +0.0933) and age midpoint (beta = +0.0777).

Scientific Limitations:
This model is an educational machine learning baseline, not a clinically approved medical device or diagnostic system. It is trained on historical EHR data (1999–2008) and reflects statistical associations rather than direct causal mechanisms. Predictions must not be used for real patient-care decisions.

GitHub Repository: https://github.com/AbhijeetArunGore/SWYNEX-Data-Science-Internship

---

### Screen-Recording Walkthrough Script (60–90 Seconds)

| Time | Visual on Screen | Spoken Script / Narration |
| :---: | :--- | :--- |
| **0:00 - 0:15** | Repository root showing clean structure (`app.py`, `src/`, `notebooks/`, `reports/`, `tests/`). | "Hello everyone! Today I'm presenting an end-to-end healthcare data science investigation: predicting 30-day diabetic hospital readmissions across 101,763 patient encounters." |
| **0:15 - 0:35** | Streamlit App Tab 4 (Exploratory Risk Patterns) showing prior inpatient visit distribution. | "In our exploratory analysis, we identified an ~8:1 class imbalance, where early readmission occurs in only 11.16% of encounters. Prior inpatient hospitalizations were the single strongest risk signal, driving readmission rates from 8.4% at zero visits up to 34.6% at 6 or more visits." |
| **0:35 - 0:55** | Streamlit App Tab 5 (Governance & Anti-Leakage) showing GroupShuffleSplit diagram. | "To enforce clinical validity, we established strict anti-leakage safeguards: first, we pruned terminal discharge outcomes that cause target leakage; second, we partitioned the data using GroupShuffleSplit on patient IDs to guarantee exactly zero patient overlap between train and test sets." |
| **0:55 - 1:15** | Streamlit App Tab 3 (Model Evaluation) showing Confusion Matrix and PR Curve. | "When evaluating on 20,093 held-out encounters, a naive majority baseline achieves 89% accuracy but catches zero patients. In contrast, our Balanced Logistic Regression successfully identifies 52.3% of all true readmissions, capturing 1,147 high-risk encounters." |
| **1:15 - 1:30** | Streamlit App Tab 2 (Risk Estimator) and Tab 5 (Limitations). | "We document all limitations transparently: this is an educational research baseline, not a certified clinical device. The code, automated tests, and reports are available on GitHub. Thank you for watching!" |
