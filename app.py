"""Diabetes Inpatient 30-Day Readmission Risk Prototype.

A professional analytical application demonstrating retrospective machine learning
for 30-day hospital readmission risk estimation using the Diabetes 130-US Hospitals dataset.

Core Capabilities:
1. Overview & Dataset Cohort Profiling
2. Interactive Risk Estimation Engine (Synthetic Scenarios)
3. Model Benchmark & Class Imbalance Evaluation
4. Exploratory Clinical Risk Patterns
5. Governance, Anti-Leakage Methodology & Model Card
"""
import os
import json
from typing import Dict, Any, Optional
import pandas as pd
import streamlit as st
from PIL import Image

from src.prediction import (
    load_model,
    validate_and_build_input_vector,
    predict_readmission_risk,
    SYNTHETIC_PROFILES,
    PredictionError,
    ModelNotFoundError,
    ModelCompatibilityError,
    InputValidationError
)

# -----------------------------------------------------------------------------
# Application Configuration & Clean Styling
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Diabetes Inpatient 30-Day Readmission Risk Prototype",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Professional neutral healthcare analytics design system
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    .disclaimer-banner {
        background-color: #fffbeb;
        border: 1px solid #fde68a;
        border-left: 5px solid #d97706;
        padding: 0.9rem 1.2rem;
        border-radius: 6px;
        margin-bottom: 1.4rem;
        font-size: 0.88rem;
        color: #92400e;
        line-height: 1.45;
    }

    .header-box {
        background-color: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 1.5rem 1.8rem;
        margin-bottom: 1.5rem;
    }

    .header-title {
        font-size: 1.8rem;
        font-weight: 700;
        color: #0f172a;
        margin: 0;
        line-height: 1.25;
    }

    .header-subtitle {
        font-size: 0.96rem;
        color: #475569;
        margin-top: 0.4rem;
        margin-bottom: 0;
        line-height: 1.5;
    }

    .kpi-container {
        display: flex;
        gap: 1rem;
        margin-bottom: 1.2rem;
    }

    .kpi-card {
        flex: 1;
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 1rem;
        text-align: center;
        box-shadow: 0 1px 3px rgba(0,0,0,0.03);
    }

    .kpi-label {
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.04em;
        color: #64748b;
    }

    .kpi-value {
        font-size: 1.6rem;
        font-weight: 700;
        color: #0f172a;
        margin-top: 2px;
    }

    .result-box-high {
        background-color: #fef2f2;
        border: 1px solid #fecaca;
        border-left: 5px solid #dc2626;
        padding: 1.2rem;
        border-radius: 8px;
        margin-bottom: 1rem;
    }

    .result-box-mod {
        background-color: #fffbeb;
        border: 1px solid #fde68a;
        border-left: 5px solid #d97706;
        padding: 1.2rem;
        border-radius: 8px;
        margin-bottom: 1rem;
    }

    .result-box-low {
        background-color: #f0fdf4;
        border: 1px solid #bbf7d0;
        border-left: 5px solid #16a34a;
        padding: 1.2rem;
        border-radius: 8px;
        margin-bottom: 1rem;
    }

    .table-container {
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        overflow: hidden;
        margin-top: 0.8rem;
    }
</style>
""", unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# Data & Model Access Layer (Cached)
# -----------------------------------------------------------------------------
MODEL_PATH = os.path.join('reports', 'modeling', 'model_pipeline.joblib')
METRICS_PATH = os.path.join('reports', 'modeling', 'metrics.json')


@st.cache_resource(show_spinner="Loading trained machine learning pipeline...")
def get_model():
    """Safely load and validate the trained pipeline."""
    try:
        return load_model(MODEL_PATH)
    except ModelNotFoundError:
        return None
    except ModelCompatibilityError as e:
        st.error(f"Model compatibility error: {str(e)}")
        return None


@st.cache_data
def get_metrics():
    """Load machine-readable test evaluation metrics."""
    if os.path.exists(METRICS_PATH):
        try:
            with open(METRICS_PATH, 'r') as f:
                return json.load(f)
        except Exception:
            return None
    return None


pipeline = get_model()
metrics_json = get_metrics()


# -----------------------------------------------------------------------------
# Header & Explicit Clinical Prototype Disclaimer
# -----------------------------------------------------------------------------
st.markdown("""
<div class="disclaimer-banner">
    <strong>Educational & Research Prototype:</strong> This application is a retrospective data science prototype
    developed using historical electronic health record data (1999–2008). It is <strong>not a clinically certified medical device</strong>,
    is not calibrated for individual patient diagnosis or triage, and must not be used for clinical decision-making.
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="header-box">
    <h1 class="header-title">Diabetes Inpatient 30-Day Readmission Risk Prototype</h1>
    <p class="header-subtitle">
        Retrospective risk estimation, imbalanced-class evaluation, and anti-leakage governance
        analyzing <strong>101,763 diabetic inpatient encounters</strong> across 130 US hospitals.
    </p>
</div>
""", unsafe_allow_html=True)

# Top KPI Summary Cards
kpi_c1, kpi_c2, kpi_c3, kpi_c4, kpi_c5 = st.columns(5)
with kpi_c1:
    st.markdown("""<div class="kpi-card"><div class="kpi-label">Analyzed Encounters</div><div class="kpi-value">101,763</div></div>""", unsafe_allow_html=True)
with kpi_c2:
    st.markdown("""<div class="kpi-card"><div class="kpi-label">Unique Patients</div><div class="kpi-value">71,515</div></div>""", unsafe_allow_html=True)
with kpi_c3:
    st.markdown("""<div class="kpi-card"><div class="kpi-label">Test Base Rate</div><div class="kpi-value">10.92%</div></div>""", unsafe_allow_html=True)
with kpi_c4:
    st.markdown("""<div class="kpi-card"><div class="kpi-label">Test Recall (50% Thresh)</div><div class="kpi-value">52.26%</div></div>""", unsafe_allow_html=True)
with kpi_c5:
    st.markdown("""<div class="kpi-card"><div class="kpi-label">Patient Split Overlap</div><div class="kpi-value">0.00%</div></div>""", unsafe_allow_html=True)

st.write("")


# -----------------------------------------------------------------------------
# Application Tabs
# -----------------------------------------------------------------------------
tab_overview, tab_estimator, tab_evaluation, tab_eda, tab_governance = st.tabs([
    "Overview & Dataset",
    "Risk Estimator",
    "Model Evaluation",
    "Exploratory Risk Patterns",
    "Governance & Model Card"
])


# =============================================================================
# TAB 1: OVERVIEW & DATASET
# =============================================================================
with tab_overview:
    st.subheader("Project Context & Dataset Provenance")
    st.markdown("""
    This project investigates 30-day early hospital readmission risk among hospitalized diabetic patients.
    Early hospital readmission represents a major operational and clinical challenge under healthcare quality programs
    (such as the US Hospital Readmissions Reduction Program). In diabetic cohorts, unplanned readmissions frequently correlate with
    glycemic volatility, acute transitional care breakdown, or compounding comorbidities.
    """)

    col_meta1, col_meta2 = st.columns(2)
    with col_meta1:
        st.markdown("#### Dataset Specifications")
        st.markdown("""
        - **Data Source**: UCI Machine Learning Repository (*Diabetes 130-US Hospitals 1999–2008*).
        - **Total Clean Encounters**: 101,763 hospital stays.
        - **Patient Population**: 71,515 unique individuals across 130 participating US clinical centers.
        - **Unit of Observation**: A single inpatient encounter lasting between 1 and 14 days.
        - **Binary Target**: Early readmission within 30 days of hospital discharge (`readmitted == '<30'`).
          - **Positive Class (`1`)**: 11,357 encounters (11.16% overall prevalence).
          - **Negative Class (`0`)**: 90,406 encounters (88.84% overall prevalence; readmission >30 days or no readmission recorded).
        """)

    with col_meta2:
        st.markdown("#### Prediction-Time Assumptions & Scope")
        st.markdown("""
        - **Intended Prediction Point**: Time of planned hospital discharge.
        - **Eligible Inputs**: Inpatient clinical and demographic variables available prior to discharge.
        - **Excluded Post-Discharge Features**: Discharge disposition codes containing terminal statuses (e.g. inpatient mortality)
          are strictly excluded to prevent artificial target leakage.
        - **Patient Clustered Structure**: Recurring patients are partitioned using strict grouped holdouts (`GroupShuffleSplit` on `patient_nbr`)
          to ensure zero patient overlap between training and test cohorts.
        """)

    st.write("---")
    st.subheader("Analytical Architecture")
    st.markdown("""
    ```
    Raw EHR Data (101,766 Encounters)
      ├── Task 1: Preprocessing & Standardization (Sentinel '?' parsing, ICD-9 mapping, 0 nulls, 101,763 clean encounters)
      ├── Task 2: Exploratory Data Analysis & Leakage Auditing (Hypothesis testing, bivariate utilization analysis)
      ├── Task 3: Patient-Grouped Holdout Split (81,670 Train / 20,093 Test; GroupShuffleSplit on patient_nbr)
      ├── Task 4: Pipeline Training (ColumnTransformer: StandardScaler + OneHotEncoder -> Balanced Logistic Regression)
      └── Evaluation: Comprehensive Imbalanced Metrics (PR-AUC, ROC-AUC, Precision, Recall, Confusion Matrix)
    ```
    """)


# =============================================================================
# TAB 2: READMISSION RISK ESTIMATOR
# =============================================================================
with tab_estimator:
    st.subheader("Interactive Readmission Risk Estimator")
    st.markdown("""
    Estimate uncalibrated relative 30-day readmission risk for a hypothetical encounter profile using the trained
    Balanced Logistic Regression pipeline.
    """)

    if pipeline is None:
        st.error(
            "Trained pipeline artifact is missing from `reports/modeling/model_pipeline.joblib`. "
            "Run `python scripts/generate_modeling_artifacts.py` to train and serialize the model."
        )
    else:
        st.markdown("##### Select a Synthetic Example Profile:")
        profile_col1, profile_col2, profile_col3 = st.columns(3)

        if "selected_profile" not in st.session_state:
            st.session_state.selected_profile = "synthetic_elevated_utilization"

        with profile_col1:
            if st.button("Synthetic Profile A (High Utilization)", use_container_width=True):
                st.session_state.selected_profile = "synthetic_elevated_utilization"
                st.rerun()
        with profile_col2:
            if st.button("Synthetic Profile B (Baseline / First-Time)", use_container_width=True):
                st.session_state.selected_profile = "synthetic_stable_baseline"
                st.rerun()
        with profile_col3:
            if st.button("Synthetic Profile C (Intermediate Risk)", use_container_width=True):
                st.session_state.selected_profile = "synthetic_intermediate_encounter"
                st.rerun()

        active_profile = SYNTHETIC_PROFILES[st.session_state.selected_profile]
        st.info(f"**Loaded Scenario:** {active_profile['label']} — {active_profile['description']}")

        st.write("---")
        input_c1, input_c2 = st.columns([1.1, 0.9])

        with input_c1:
            st.markdown("##### Clinical Encounter Parameters")
            vals = active_profile['inputs']

            sub_c1, sub_c2 = st.columns(2)
            with sub_c1:
                age_bracket = st.selectbox(
                    "Age Bracket",
                    ["[0-10)", "[10-20)", "[20-30)", "[30-40)", "[40-50)", "[50-60)", "[60-70)", "[70-80)", "[80-90)", "[90-100)"],
                    index=["[0-10)", "[10-20)", "[20-30)", "[30-40)", "[40-50)", "[50-60)", "[60-70)", "[70-80)", "[80-90)", "[90-100)"].index(vals.get('age_bracket', '[60-70)'))
                )
                gender = st.selectbox("Gender", ["Female", "Male"], index=0 if vals.get('gender') == 'Female' else 1)
                admission_type_desc = st.selectbox(
                    "Admission Urgency",
                    ["Emergency", "Urgent", "Elective", "Not Available / Unmapped"],
                    index=["Emergency", "Urgent", "Elective", "Not Available / Unmapped"].index(vals.get('admission_type_desc', 'Emergency'))
                )
                number_inpatient = st.slider("Prior Inpatient Admissions (12 Months)", 0, 15, int(vals.get('number_inpatient', 0)))
                number_emergency = st.slider("Prior Emergency Visits (12 Months)", 0, 10, int(vals.get('number_emergency', 0)))
                time_in_hospital = st.slider("Length of Stay (Days)", 1, 14, int(vals.get('time_in_hospital', 4)))

            with sub_c2:
                number_diagnoses = st.slider("Number of Diagnoses", 1, 16, int(vals.get('number_diagnoses', 5)))
                diag_1_category = st.selectbox(
                    "Primary Diagnosis Category (ICD-9)",
                    ["Circulatory", "Diabetes", "Respiratory", "Digestive", "Genitourinary", "Musculoskeletal", "Neoplasms", "Injury", "Other"],
                    index=["Circulatory", "Diabetes", "Respiratory", "Digestive", "Genitourinary", "Musculoskeletal", "Neoplasms", "Injury", "Other"].index(vals.get('diag_1_category', 'Circulatory'))
                )
                A1Cresult = st.selectbox("HbA1c Result", ["Not Tested", "Norm", ">7", ">8"], index=["Not Tested", "Norm", ">7", ">8"].index(vals.get('A1Cresult', 'Not Tested')))
                change = st.selectbox("Medication Dosage Changed?", ["Ch", "No"], index=["Ch", "No"].index(vals.get('change', 'No')))
                insulin = st.selectbox("Insulin Status", ["No", "Steady", "Up", "Down"], index=["No", "Steady", "Up", "Down"].index(vals.get('insulin', 'No')))
                num_medications = st.slider("Distinct Medications", 1, 50, int(vals.get('num_medications', 10)))

        # Run Prediction
        user_dict = {
            'age_bracket': age_bracket,
            'gender': gender,
            'race': vals.get('race', 'Unknown'),
            'admission_type_desc': admission_type_desc,
            'admission_source_desc': vals.get('admission_source_desc', 'Emergency Room'),
            'time_in_hospital': time_in_hospital,
            'num_lab_procedures': vals.get('num_lab_procedures', 40),
            'num_procedures': vals.get('num_procedures', 1),
            'num_medications': num_medications,
            'number_outpatient': vals.get('number_outpatient', 0),
            'number_emergency': number_emergency,
            'number_inpatient': number_inpatient,
            'number_diagnoses': number_diagnoses,
            'diag_1_category': diag_1_category,
            'diag_2_category': vals.get('diag_2_category', 'Other'),
            'diag_3_category': vals.get('diag_3_category', 'Other'),
            'max_glu_serum': vals.get('max_glu_serum', 'Not Tested'),
            'A1Cresult': A1Cresult,
            'change': change,
            'diabetesMed': vals.get('diabetesMed', 'Yes'),
            'insulin': insulin,
            'medical_specialty': vals.get('medical_specialty', 'InternalMedicine'),
            'payer_code': vals.get('payer_code', 'Unknown')
        }

        try:
            input_vector = validate_and_build_input_vector(user_dict)
            pred_results = predict_readmission_risk(pipeline, input_vector, decision_threshold=0.50)
        except InputValidationError as e:
            st.error(f"Input validation error: {str(e)}")
            pred_results = None
        except PredictionError as e:
            st.error(f"Prediction execution error: {str(e)}")
            pred_results = None

        with input_c2:
            st.markdown("##### Risk Estimation Output")
            if pred_results:
                tier = pred_results['tier_level']
                box_class = (
                    "result-box-high" if tier == "high"
                    else "result-box-mod" if tier == "moderate"
                    else "result-box-low"
                )

                st.markdown(f"""
                <div class="{box_class}">
                    <div style="font-size: 0.8rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.05em;">
                        {pred_results['risk_tier']}
                    </div>
                    <div style="font-size: 2.4rem; font-weight: 800; margin: 4px 0;">
                        {pred_results['probability_percent']}%
                    </div>
                    <div style="font-size: 0.88rem; color: #334155;">
                        Estimated Statistical Risk of 30-Day Readmission
                    </div>
                    <hr style="margin: 8px 0; border: none; border-top: 1px solid rgba(0,0,0,0.1);"/>
                    <div style="font-size: 0.82rem; line-height: 1.45;">
                        {pred_results['tier_description']}
                    </div>
                </div>
                """, unsafe_allow_html=True)

                st.warning(f"ℹ️ **Calibration Note**: {pred_results['calibration_note']}")

                st.markdown("###### Primary Model Feature Factors for this Profile:")
                for factor in pred_results['risk_factors']:
                    st.markdown(f"- {factor}")


# =============================================================================
# TAB 3: MODEL EVALUATION & BENCHMARK
# =============================================================================
with tab_evaluation:
    st.subheader("Model Evaluation & Imbalanced-Class Benchmark")
    st.markdown("""
    Evaluation was conducted on a held-out test cohort of **20,093 encounters** (14,303 unique patients)
    partitioned using `GroupShuffleSplit` on `patient_nbr` ensuring **zero patient overlap** with the training set.
    """)

    st.markdown("#### The Accuracy Trap in Healthcare Classification")
    st.markdown("""
    In imbalanced classification (10.92% test prevalence, ~8:1 negative-to-positive ratio),
    **accuracy is a misleading metric**. A trivial baseline that guesses "no readmission" for every patient achieves
    **89.08% accuracy**, yet fails to identify a single true positive (0.0% recall).
    """)

    # Metrics Benchmark Table
    metrics_table = pd.DataFrame([
        {
            "Model Architecture": "DummyClassifier (Majority Class)",
            "Decision Threshold": "N/A (Always 0)",
            "Accuracy": "89.08%",
            "Precision": "0.00%",
            "Recall (Sensitivity)": "0.00% (0 / 2,195)",
            "F1-Score": "0.0000",
            "ROC-AUC": "0.5000",
            "PR-AUC (Avg Precision)": "0.1092 (Base Rate)"
        },
        {
            "Model Architecture": "Logistic Regression (Balanced Weights)",
            "Decision Threshold": "0.50",
            "Accuracy": "65.86%",
            "Precision": "16.48%",
            "Recall (Sensitivity)": "52.26% (1,147 / 2,195)",
            "F1-Score": "0.2506",
            "ROC-AUC": "0.6435",
            "PR-AUC (Avg Precision)": "0.1887 (+72.8% rel. gain)"
        }
    ])
    st.dataframe(metrics_table, hide_index=True, use_container_width=True)

    st.write("---")
    st.subheader("Held-Out Test Confusion Matrix Breakdown (N = 20,093)")

    cm_c1, cm_c2, cm_c3, cm_c4 = st.columns(4)
    with cm_c1:
        st.metric("True Negatives (TN)", "12,086", help="Correctly identified non-readmissions; unnecessary follow-up costs avoided.")
    with cm_c2:
        st.metric("False Positives (FP)", "5,812", help="Encounters flagged as high risk that did not readmit within 30 days.")
    with cm_c3:
        st.metric("False Negatives (FN)", "1,048", help="Encounters that resulted in 30-day readmissions but were missed by the linear decision boundary.")
    with cm_c4:
        st.metric("True Positives (TP)", "1,147", help="Readmission encounters successfully identified.")

    st.markdown("""
    > **Operational Trade-Off Discussion**:
    > With an uncalibrated linear model under class-balanced weighting, the decision boundary trades specificity for sensitivity.
    > Capturing 52.3% of early readmissions (1,147 encounters) requires accepting 5,812 false positives. In hospital operations,
    > the clinical penalty of an unexpected deterioration (false negative) is typically substantial, while a false positive leads to
    > non-invasive preventive outreach (such as nurse phone check-ins).
    """)

    st.write("---")
    st.subheader("Diagnostic Curves & Validation Visualizations")
    img_col1, img_col2 = st.columns(2)
    roc_fig_path = os.path.join("reports", "modeling", "figures", "04_precision_recall_and_roc_curves.png")
    cm_fig_path = os.path.join("reports", "modeling", "figures", "02_confusion_matrix_comparison.png")

    with img_col1:
        if os.path.exists(roc_fig_path):
            st.image(roc_fig_path, caption="Precision-Recall Curve (PR-AUC: 0.1887 vs 0.1092 base rate) & ROC Curve (ROC-AUC: 0.6435)")
    with img_col2:
        if os.path.exists(cm_fig_path):
            st.image(cm_fig_path, caption="Confusion Matrix Comparison: Baseline vs Balanced Logistic Regression")


# =============================================================================
# TAB 4: EXPLORATORY RISK PATTERNS
# =============================================================================
with tab_eda:
    st.subheader("Exploratory Data Analysis & Empirical Risk Drivers")
    st.markdown("""
    Statistical tests across the 101,763 hospital encounters identified significant empirical predictors
    associated with early readmission:
    """)

    eda_p1, eda_p2 = st.columns(2)
    with eda_p1:
        st.markdown("""
        ##### 1. Prior Inpatient Admissions ($p < 10^{-15}$)
        - Encounters with **0 prior hospitalizations** had an early readmission rate of **8.44%**.
        - Risk rises monotonically: **12.92%** at 1 visit, **20.29%** at 3 visits, and **34.58%** at 6+ visits.
        - **Mann-Whitney U Test**: $U = 401,064,764, p < 10^{-15}$ (Statistically significant difference between cohorts).
        - In multivariable regression, `number_inpatient` exhibits the highest positive coefficient ($\beta = +0.3517$).
        """)
        fig_inp = os.path.join("reports", "eda", "figures", "03_prior_inpatient_utilization.png")
        if os.path.exists(fig_inp):
            st.image(fig_inp, caption="Figure: 30-Day Readmission Rate Escalation by Prior Inpatient Admissions")

    with eda_p2:
        st.markdown(r"""
        ##### 2. Medication Changes & Glycemic Titration ($p < 10^{-46}$)
        - Encounters where diabetic medications were titrated (`change == 'Ch'`) experienced a **12.24%** readmission rate
          compared to **10.23%** for stable regimens.
        - **Pearson Chi-Square Test**: $\chi^2 = 215.94, p = 1.289 \times 10^{-47}$ (Cramer's $V = 0.0461$).
        - Reflects that in-hospital medication adjustment is an observable indicator of unstable glycemic progression.
        """)
        fig_med = os.path.join("reports", "eda", "figures", "06_hba1c_and_medchange.png")
        if os.path.exists(fig_med):
            st.image(fig_med, caption="Figure: Bivariate Risk by Medication Titration & HbA1c Status")

    st.write("---")
    st.subheader("Standardized Model Feature Coefficients")
    st.markdown("""
    Standardized log-odds coefficients from the multivariable logistic regression model fitted on the training split:
    """)
    fig_coef = os.path.join("reports", "modeling", "figures", "05_top_feature_coefficients.png")
    if os.path.exists(fig_coef):
        st.image(fig_coef, caption="Figure: Standardized Logistic Regression Coefficients for Early Hospital Readmission")


# =============================================================================
# TAB 5: GOVERNANCE & MODEL CARD
# =============================================================================
with tab_governance:
    st.subheader("Methodological Governance & Model Card")
    st.markdown("""
    To ensure scientific rigor, anti-leakage rules and explicit limitations were documented following standard
    machine learning reporting guidelines.
    """)

    g_c1, g_c2, g_c3 = st.columns(3)
    with g_c1:
        st.markdown("""
        ##### 1. Patient-Grouped Holdout Split
        - **Threat**: 71,515 unique patients generate 101,763 encounters. A naive random split places repeated encounters
          from the same individual into both training and test sets.
        - **Mitigation**: Used `GroupShuffleSplit` on `patient_nbr` (81,670 train / 20,093 test).
        - **Result**: Exactly **zero patient overlap** between splits, preventing optimistic evaluation leakage.
        """)
    with g_c2:
        st.markdown("""
        ##### 2. Leakage Feature Exclusion
        - **Threat**: Post-discharge events such as patient mortality (`discharge_disposition` containing `Expired`)
          produce 0% readmissions because deceased patients cannot be readmitted.
        - **Mitigation**: All `discharge_disposition` features were excluded prior to model fitting.
        - **Result**: The model uses only pre-discharge clinical information.
        """)
    with g_c3:
        st.markdown("""
        ##### 3. Preprocessing Encapsulation
        - **Threat**: Fitting imputation or scaling transforms on the whole dataset leaks test distribution parameters.
        - **Mitigation**: Standard scaling and one-hot encoding are encapsulated in a Scikit-Learn `Pipeline` fitted
          strictly on the training split.
        - **Result**: Complete isolation of held-out test data.
        """)

    st.write("---")
    st.markdown("#### Transparent Limitations & Out-of-Scope Uses")
    st.markdown("""
    1. **Out-of-Scope Clinical Use**: This tool is not a diagnostic device, clinical decision support system, or hospital triage protocol.
       It must not be used to withhold care, alter medical prescriptions, or dictate clinical admission/discharge decisions.
    2. **Historical Observational Scope**: Data was collected between 1999 and 2008. It does not reflect modern clinical practices,
       such as continuous glucose monitoring (CGM) or contemporary pharmacotherapies (e.g. SGLT2 inhibitors or GLP-1 receptor agonists).
    3. **Association vs. Causation**: Regression coefficients quantify observational statistical correlations in tabular EHR data;
       they do not prove causal clinical mechanisms.
    4. **Uncalibrated Output**: Predicted probabilities reflect raw logistic outputs under balanced weighting,
       serving as relative risk rankings rather than true calibrated clinical probabilities.
    """)

    st.write("---")
    st.markdown("#### Documentation References")
    st.markdown("""
    - Detailed Model Card: `docs/model_card.md`
    - Feature Schema & Dictionary: `docs/data_dictionary.md`
    - Security & Data Governance Policy: `docs/security.md`
    - Capstone Analytical Report: `reports/final_case_study_report.md`
    """)


# -----------------------------------------------------------------------------
# Neutral Professional Footer
# -----------------------------------------------------------------------------
st.write("---")
st.markdown("""
<div style="text-align: center; color: #64748b; font-size: 0.82rem; padding: 0.8rem 0;">
    Diabetes Inpatient 30-Day Readmission Risk Prototype • Educational & Research Benchmark<br/>
    Repository: <code>SWYNEX-Data-Science-Internship</code> • Python 3.12 • Scikit-Learn 1.6 • Streamlit
</div>
""", unsafe_allow_html=True)
