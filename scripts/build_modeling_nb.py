import os
import nbformat as nbf

nb_path = os.path.join('notebooks', '03_modeling_and_evaluation.ipynb')
with open(nb_path, 'r', encoding='utf-8') as f:
    nb = nbf.read(f, as_version=4)

m9 = """---
## Section 9: Interpret the Results

### 1. Does Logistic Regression Improve on the Dummy Baseline?
**Yes, substantially.** While the DummyClassifier achieved an apparent accuracy of 89.08%, it had a **Recall of 0.0%**—meaning it failed to identify even a single readmitted patient.

Our Balanced Logistic Regression model achieves:
- **Recall of 52.26%**: The model successfully detects **1,147 out of 2,195** true 30-day readmissions on the unseen test set.
- **PR-AUC of 0.1887**: A **+72.8% relative improvement** over the baseline positive rate (0.1092).
- **ROC-AUC of 0.6435**: Significantly better than random chance (0.5000).

### 2. The Precision-Recall Trade-Off in Plain English
Because early readmissions are relatively uncommon (10.92% of the test cohort), our use of `class_weight='balanced'` prioritizes sensitivity (catching positive cases) over precision.

- **Precision is 16.48%**: When the model sounds an alarm and flags an encounter as high-risk, approximately **1 in 6** of those patients will actually be readmitted within 30 days.
- **Operational Reality**: Out of 6,959 total positive predictions made by the model, 1,147 are True Positives and 5,812 are False Positives. In hospital operations, a false alarm typically results in a low-cost, harmless intervention (such as a pharmacist reviewing discharge prescriptions or a follow-up phone call from a transitional care nurse). In contrast, a False Negative (missing a patient who deteriorates at home) can lead to emergency hospitalization or death.

### 3. Understanding the Top Feature Drivers
Looking at the model's coefficients:
- **`number_inpatient` ($\beta = +0.3517$)**: The strongest positive numerical feature. Having prior hospitalizations in the preceding year substantially increases the log-odds of early readmission.
- **`number_diagnoses` ($\beta = +0.0933$) and `age_midpoint` ($\beta = +0.0777$)**: Captures chronic multi-morbidity and geriatric frailty.
- **Specialty Flags**: Inpatient encounters managed by `Allergy and Immunology` (+1.03) and `Hematology` (+0.98) have higher readmission probabilities, likely reflecting complex systemic immune and oncological disorders.
"""

m10 = """---
## Section 10: Limitations

An honest data scientist must clearly state the limitations of their work:

1. **Historical Observational Dataset (1999–2008)**: Clinical guidelines, medications, and electronic health records have advanced significantly since 2008 (e.g. widespread modern continuous glucose monitors and GLP-1 receptor agonists). The model should not be assumed to generalize to modern patient populations without retraining.
2. **Recorded vs. Unrecorded Readmissions**: The target variable captures readmissions recorded within the monitored hospital system. If a patient was admitted to an independent, unmonitored hospital, that outcome was recorded as `NO` in this dataset.
3. **Class Imbalance**: An 8:1 class ratio naturally depresses precision. Achieving high precision while maintaining high recall is difficult with a linear model on clinical tabular data.
4. **Baseline Model Simplicity**: A single Logistic Regression with linear decision boundaries cannot capture non-linear feature interactions (for example, the compounding risk of prior inpatient visits combined with specific high-risk medication changes).
5. **Feature Availability Timing**: For our model to be used during the hospital stay, we deliberately excluded post-discharge destination (`discharge_disposition_desc`). Features must always be aligned with the operational moment when the prediction is actually needed.
6. **Correlation vs. Causation**: Regression coefficients describe empirical associations in observational data. They do not prove that altering any specific feature will reduce a patient's readmission risk.
"""

m11 = """---
## Section 11: Conclusion and Next Steps

### What Was Built:
- An end-to-end, leak-free machine learning pipeline using scikit-learn's `ColumnTransformer`, `StandardScaler`, `OneHotEncoder`, and `LogisticRegression`.
- A leak-free, patient-grouped evaluation split (`GroupShuffleSplit` on `patient_nbr`) ensuring zero patient overlap between train and test partitions.
- Deliberate exclusion of identifiers (`encounter_id`, `patient_nbr`) and terminal status leakage features (`discharge_disposition`).

### What We Learned:
- Evaluating imbalanced healthcare models using accuracy alone is dangerous and misleading.
- A balanced Logistic Regression baseline provides genuine predictive utility, recovering **52.26% of true 30-day early readmissions** and achieving a **PR-AUC of 0.1887 (+72.8% over baseline)**.
- Historical hospital admissions (`number_inpatient`) and clinical multi-morbidity (`number_diagnoses`) serve as the strongest positive risk indicators.

### Potential Improvements for Future Iterations:
- **Tree-Based Ensembles**: Evaluate gradient boosting models (such as LightGBM or XGBoost) to test whether non-linear decision trees capture complex clinical interactions.
- **Threshold Tuning**: Move beyond the default 0.50 probability cutoff to align decision thresholds with specific hospital nurse capacity and budget constraints.
- **Feature Engineering**: Construct composite ratio features, such as inpatient utilization velocity (`number_inpatient / (number_outpatient + 1)`).

---
**Task 3 Status**: Complete, reproducible, and ready for internship review!
"""

nb.cells.extend([
    nbf.v4.new_markdown_cell(m9),
    nbf.v4.new_markdown_cell(m10),
    nbf.v4.new_markdown_cell(m11)
])

with open(nb_path, 'w', encoding='utf-8') as f:
    nbf.write(nb, f)

print(f"Updated {nb_path} with Sections 9 to 11. Notebook assembly complete!")
