# SWYNEX Internship — Task 3: Modeling and Evaluation
## LinkedIn Post Copy & Video Walkthrough Guide

---

### 📝 LinkedIn Post Draft

Excited to share that I have completed **Task 3: Modeling and Evaluation** for my Data Science Internship with @SWYNEX Technologies! 🚀

Building upon the clean clinical foundation from Task 1 and the empirical findings from Task 2, this project focuses on a high-stakes clinical challenge: predicting **30-day early hospital readmissions** for diabetic inpatients using the **Diabetes 130-US Hospitals dataset (1999–2008)**.

🎯 **The Prediction Problem**:
- **Target**: Binary outcome where Positive (`1`) represents early readmission within 30 days (`<30`), and Negative (`0`) represents no recorded 30-day readmission (`NO` or `>30`).
- **Base Rate**: Early readmission is an uncommon outcome, occurring in only **11.16%** of all encounters (and **10.92%** of our held-out test cohort).

🛡️ **Anti-Leakage & Governance First**:
- **Patient-Grouped Split**: Rather than a naive random split, we implemented `GroupShuffleSplit` on `patient_nbr` (81,670 train / 20,093 test). This strictly ensures **0 patient overlap** between training and test sets, avoiding optimistic data leakage from recurring patients.
- **Leakage Feature Pruning**: Deliberately excluded hospital discharge destinations (`discharge_disposition`) containing post-event terminal statuses (like patient mortality), preserving only features available during the hospital encounter.

📊 **Baseline vs. Balanced Logistic Regression (Held-Out Test Set, N = 20,093)**:
1. **The "Accuracy Trap" of the Dummy Baseline**:
   - A majority-class `DummyClassifier` achieves **89.08% accuracy**, yet has a **Recall of 0.00%** and **F1 of 0.00%**—completely failing to detect all 2,195 readmitted patients.
2. **Balanced Logistic Regression Results**:
   - **Recall (Sensitivity)**: **52.26%** — Successfully identifies **1,147 out of 2,195** true 30-day readmissions.
   - **PR-AUC (Average Precision)**: **0.1887** — A **+72.8% relative improvement** over the baseline test prevalence (0.1092).
   - **ROC-AUC**: **0.6435** — Demonstrates solid discriminative power above random chance (0.5000).
   - **Precision**: **16.48%** — When the model flags a patient as high-risk, ~1 in 6 is readmitted within 30 days.

💡 **Key Takeaway & Clinical Context**:
In healthcare operations, evaluating imbalanced models using accuracy alone is dangerous. Using `class_weight='balanced'` accepts a higher False Positive rate (5,812 alerts) to avoid missing patients who need transitional support. In a hospital setting, a false alarm triggers a low-cost preventive check (such as a follow-up call from a transitional care nurse), whereas a False Negative can mean an unmonitored patient suffering severe complications at home.

⚠️ **Limitations & Scientific Honesty**:
This model is an educational machine learning baseline, **not a clinically validated diagnostic tool**. It is trained on historical observational data (1999–2008) and relies on linear decision boundaries. Logistic regression coefficients represent statistical associations, not causal relationships.

🛠️ **Technologies**: Python 3.12, Scikit-learn (`Pipeline`, `ColumnTransformer`, `LogisticRegression`, `GroupShuffleSplit`), Pandas, NumPy, Matplotlib, Seaborn, Pytest.

📁 **GitHub Repository**: https://github.com/AbhijeetArunGore/SWYNEX-Data-Preparation  
📓 **Task 3 Notebook**: `notebooks/03_modeling_and_evaluation.ipynb`  
📄 **Executive Evaluation Report**: `reports/modeling/model_evaluation_report.md`  
📈 **Metrics Artifact**: `reports/modeling/metrics.json`  

#SWYNEX #Internship #DataScience #MachineLearning #HealthcareAnalytics #Python #ScikitLearn #ModelEvaluation #ClinicalData #PredictiveModeling

---

### 🎥 60–90 Second Screen Recording Video Walkthrough Script

| Time | Visual on Screen | Spoken Script / Narration |
| :---: | :--- | :--- |
| **0:00 - 0:15** | GitHub repository root ([SWYNEX-Data-Preparation](https://github.com/AbhijeetArunGore/SWYNEX-Data-Preparation)), showing `notebooks/03_modeling_and_evaluation.ipynb`, `reports/modeling/`, and `src/modeling.py`. | "Hello everyone! Today I'm excited to present Task 3 of my Data Science internship at SWYNEX Technologies: Modeling and Evaluation for 30-day diabetic hospital readmissions." |
| **0:15 - 0:35** | Scroll to Section 3 and Section 4 in `notebooks/03_modeling_and_evaluation.ipynb`, highlighting feature selection and `GroupShuffleSplit`. | "To build a realistic clinical model, we implemented two critical anti-leakage safeguards: first, we excluded post-discharge terminal status features; second, we partitioned our 101,763 encounters using GroupShuffleSplit on patient IDs to guarantee zero patient overlap between train and test sets." |
| **0:35 - 0:55** | Highlight Section 5 (DummyClassifier) and Section 7 (Logistic Regression classification reports and confusion matrix heatmap). | "When evaluating our baseline DummyClassifier, it achieved an 89.08% accuracy—yet had a recall of 0%, missing every single positive case. In contrast, our Balanced Logistic Regression caught 52.26% of all 30-day readmissions, recovering 1,147 high-risk patients on the held-out test cohort." |
| **0:55 - 1:15** | Show Section 8 figures: PR Curve (`04_precision_recall_and_roc_curves.png`) and Top Feature Coefficients (`05_top_feature_coefficients.png`). | "Our Precision-Recall AUC reached 0.1887, representing a 72.8% relative improvement over baseline prevalence. Top model drivers confirmed that prior inpatient admissions and multi-morbidity are the strongest empirical predictors of readmission risk." |
| **1:15 - 1:30** | Show `reports/modeling/model_evaluation_report.md` and repository README, pointing out the limitations section. | "We also clearly document that this is a baseline research model and not clinically validated for hospital deployment. All code, 17 unit tests, and executive reports are open-source on GitHub. Thank you for following my SWYNEX internship journey!" |
