"""Prediction and inference engine for diabetic inpatient readmission risk.

Provides schema validation, input verification, safe model loading,
and probability estimation with explicit disclaimers for educational/analytical use.
"""
from typing import Any, Dict, List, Optional, Tuple
import os
import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline


class PredictionError(Exception):
    """Base exception for prediction module errors."""
    pass


class ModelNotFoundError(PredictionError):
    """Raised when the serialized model artifact cannot be found."""
    pass


class ModelCompatibilityError(PredictionError):
    """Raised when the loaded model does not match expected schema or pipeline structure."""
    pass


class InputValidationError(PredictionError):
    """Raised when user-provided input features fall outside valid ranges or allowed categories."""
    pass


# Expected numerical features and their valid physiological/administrative ranges
NUMERICAL_FEATURE_RANGES: Dict[str, Tuple[float, float]] = {
    'time_in_hospital': (1.0, 14.0),
    'num_lab_procedures': (1.0, 150.0),
    'num_procedures': (0.0, 10.0),
    'num_medications': (1.0, 100.0),
    'number_outpatient': (0.0, 50.0),
    'number_emergency': (0.0, 50.0),
    'number_inpatient': (0.0, 30.0),
    'number_diagnoses': (1.0, 20.0),
    'age_midpoint': (5.0, 95.0),
}

# The 34 categorical feature names used in the leak-free feature matrix
EXPECTED_CATEGORICAL_FEATURES: List[str] = [
    'race', 'gender', 'payer_code', 'medical_specialty',
    'max_glu_serum', 'A1Cresult',
    'metformin', 'repaglinide', 'nateglinide', 'chlorpropamide',
    'glimepiride', 'acetohexamide', 'glipizide', 'glyburide',
    'tolbutamide', 'pioglitazone', 'rosiglitazone', 'acarbose',
    'miglitol', 'troglitazone', 'tolazamide', 'insulin',
    'glyburide-metformin', 'glipizide-metformin',
    'glimepiride-pioglitazone', 'metformin-rosiglitazone',
    'metformin-pioglitazone', 'change', 'diabetesMed',
    'admission_type_desc', 'admission_source_desc',
    'diag_1_category', 'diag_2_category', 'diag_3_category'
]

# All 43 expected input features in deterministic order
EXPECTED_INPUT_COLUMNS: List[str] = list(NUMERICAL_FEATURE_RANGES.keys()) + EXPECTED_CATEGORICAL_FEATURES

# Standard age bracket to numeric midpoint conversion
AGE_BRACKET_MAP: Dict[str, float] = {
    '[0-10)': 5.0,
    '[10-20)': 15.0,
    '[20-30)': 25.0,
    '[30-40)': 35.0,
    '[40-50)': 45.0,
    '[50-60)': 55.0,
    '[60-70)': 65.0,
    '[70-80)': 75.0,
    '[80-90)': 85.0,
    '[90-100)': 95.0,
}

# Synthetic scenario profiles for educational/validation demonstration
SYNTHETIC_PROFILES: Dict[str, Dict[str, Any]] = {
    'synthetic_elevated_utilization': {
        'label': 'Synthetic Profile A — High Historical Inpatient Utilization',
        'description': (
            'Hypothetical scenario: Elderly encounter with 4 prior inpatient admissions, '
            'multi-morbidity (9 recorded diagnoses), active inpatient medication titration, '
            'and elevated HbA1c.'
        ),
        'inputs': {
            'age_bracket': '[70-80)',
            'gender': 'Female',
            'race': 'Caucasian',
            'admission_type_desc': 'Emergency',
            'admission_source_desc': 'Emergency Room',
            'time_in_hospital': 8,
            'num_lab_procedures': 58,
            'num_procedures': 1,
            'num_medications': 22,
            'number_outpatient': 1,
            'number_emergency': 2,
            'number_inpatient': 4,
            'number_diagnoses': 9,
            'diag_1_category': 'Circulatory',
            'diag_2_category': 'Diabetes',
            'diag_3_category': 'Respiratory',
            'max_glu_serum': 'Not Tested',
            'A1Cresult': '>8',
            'change': 'Ch',
            'diabetesMed': 'Yes',
            'insulin': 'Up',
            'medical_specialty': 'InternalMedicine',
            'payer_code': 'Unknown'
        }
    },
    'synthetic_stable_baseline': {
        'label': 'Synthetic Profile B — Baseline / First-Time Encounter',
        'description': (
            'Hypothetical scenario: Younger adult encounter with 0 prior inpatient stays, '
            'short length of stay (2 days), stable medication regimen, and controlled HbA1c.'
        ),
        'inputs': {
            'age_bracket': '[30-40)',
            'gender': 'Male',
            'race': 'Caucasian',
            'admission_type_desc': 'Elective',
            'admission_source_desc': 'Physician Referral',
            'time_in_hospital': 2,
            'num_lab_procedures': 25,
            'num_procedures': 1,
            'num_medications': 7,
            'number_outpatient': 0,
            'number_emergency': 0,
            'number_inpatient': 0,
            'number_diagnoses': 3,
            'diag_1_category': 'Diabetes',
            'diag_2_category': 'Other',
            'diag_3_category': 'Other',
            'max_glu_serum': 'Not Tested',
            'A1Cresult': 'Norm',
            'change': 'No',
            'diabetesMed': 'Yes',
            'insulin': 'No',
            'medical_specialty': 'InternalMedicine',
            'payer_code': 'Unknown'
        }
    },
    'synthetic_intermediate_encounter': {
        'label': 'Synthetic Profile C — Intermediate Risk / Single Prior Admission',
        'description': (
            'Hypothetical scenario: Intermediate age encounter with 1 prior inpatient admission, '
            'moderate length of stay (4 days), and respiratory primary diagnosis.'
        ),
        'inputs': {
            'age_bracket': '[60-70)',
            'gender': 'Male',
            'race': 'AfricanAmerican',
            'admission_type_desc': 'Emergency',
            'admission_source_desc': 'Emergency Room',
            'time_in_hospital': 4,
            'num_lab_procedures': 42,
            'num_procedures': 0,
            'num_medications': 14,
            'number_outpatient': 2,
            'number_emergency': 0,
            'number_inpatient': 1,
            'number_diagnoses': 6,
            'diag_1_category': 'Respiratory',
            'diag_2_category': 'Circulatory',
            'diag_3_category': 'Diabetes',
            'max_glu_serum': 'Not Tested',
            'A1Cresult': 'Not Tested',
            'change': 'No',
            'diabetesMed': 'Yes',
            'insulin': 'Steady',
            'medical_specialty': 'InternalMedicine',
            'payer_code': 'Unknown'
        }
    }
}


def load_model(model_path: str = os.path.join('reports', 'modeling', 'model_pipeline.joblib')) -> Pipeline:
    """Load and validate a trained scikit-learn model pipeline.

    Parameters
    ----------
    model_path : str
        Relative or absolute path to the serialized .joblib pipeline file.

    Returns
    -------
    Pipeline
        Validated scikit-learn Pipeline object.

    Raises
    ------
    ModelNotFoundError
        If the file does not exist at the specified path.
    ModelCompatibilityError
        If the loaded artifact is not a Pipeline or lacks required steps ('prep', 'clf').
    """
    if not os.path.exists(model_path):
        raise ModelNotFoundError(
            f"Model artifact not found at '{model_path}'. "
            "Please generate the model using: python scripts/generate_modeling_artifacts.py"
        )

    try:
        model = joblib.load(model_path)
    except Exception as e:
        raise ModelCompatibilityError(f"Failed to deserialize model file at '{model_path}': {str(e)}") from e

    if not isinstance(model, Pipeline):
        raise ModelCompatibilityError(f"Expected a scikit-learn Pipeline, got {type(model).__name__}.")

    step_names = list(model.named_steps.keys())
    if 'prep' not in step_names or 'clf' not in step_names:
        raise ModelCompatibilityError(
            f"Pipeline missing required steps 'prep' and 'clf'. Found steps: {step_names}."
        )

    return model


def validate_and_build_input_vector(
    user_inputs: Dict[str, Any],
    reference_df: Optional[pd.DataFrame] = None
) -> pd.DataFrame:
    """Validate user inputs and construct a single-row feature DataFrame.

    Parameters
    ----------
    user_inputs : Dict[str, Any]
        Raw input dictionary containing clinical and encounter variables.
    reference_df : Optional[pd.DataFrame], default=None
        Sample DataFrame used to populate default categorical values for non-interactive columns.

    Returns
    -------
    pd.DataFrame
        Single-row DataFrame matching the exact feature matrix schema.

    Raises
    ------
    InputValidationError
        If numerical inputs are out of valid range or required fields are missing.
    """
    input_data: Dict[str, Any] = {}

    # 1. Parse age midpoint
    if 'age_midpoint' in user_inputs:
        age_val = float(user_inputs['age_midpoint'])
    elif 'age_bracket' in user_inputs:
        bracket = str(user_inputs['age_bracket']).strip()
        if bracket not in AGE_BRACKET_MAP:
            raise InputValidationError(f"Invalid age bracket: '{bracket}'. Allowed: {list(AGE_BRACKET_MAP.keys())}")
        age_val = AGE_BRACKET_MAP[bracket]
    else:
        age_val = 65.0
    input_data['age_midpoint'] = age_val

    # 2. Validate numerical features
    for num_col, (min_val, max_val) in NUMERICAL_FEATURE_RANGES.items():
        if num_col == 'age_midpoint':
            continue
        val = user_inputs.get(num_col)
        if val is None:
            raise InputValidationError(f"Missing required numerical feature: '{num_col}'.")
        try:
            val_f = float(val)
        except (ValueError, TypeError) as e:
            raise InputValidationError(f"Invalid numeric value for '{num_col}': {val}.") from e

        if not (min_val <= val_f <= max_val):
            raise InputValidationError(
                f"Feature '{num_col}' value {val_f} is outside acceptable range [{min_val}, {max_val}]."
            )
        input_data[num_col] = val_f

    # 3. Populate default categoricals
    # Default unselected diabetes medications to 'No'
    med_columns = [
        'metformin', 'repaglinide', 'nateglinide', 'chlorpropamide',
        'glimepiride', 'acetohexamide', 'glipizide', 'glyburide',
        'tolbutamide', 'pioglitazone', 'rosiglitazone', 'acarbose',
        'miglitol', 'troglitazone', 'tolazamide', 'insulin',
        'glyburide-metformin', 'glipizide-metformin',
        'glimepiride-pioglitazone', 'metformin-rosiglitazone',
        'metformin-pioglitazone'
    ]
    for med in med_columns:
        input_data[med] = str(user_inputs.get(med, 'No'))

    # Administrative and clinical categoricals
    input_data['race'] = str(user_inputs.get('race', 'Unknown'))
    input_data['gender'] = str(user_inputs.get('gender', 'Female'))
    input_data['payer_code'] = str(user_inputs.get('payer_code', 'Unknown'))
    input_data['medical_specialty'] = str(user_inputs.get('medical_specialty', 'InternalMedicine'))
    input_data['max_glu_serum'] = str(user_inputs.get('max_glu_serum', 'Not Tested'))
    input_data['A1Cresult'] = str(user_inputs.get('A1Cresult', 'Not Tested'))
    input_data['change'] = str(user_inputs.get('change', 'No'))
    input_data['diabetesMed'] = str(user_inputs.get('diabetesMed', 'Yes'))
    input_data['admission_type_desc'] = str(user_inputs.get('admission_type_desc', 'Emergency'))
    input_data['admission_source_desc'] = str(user_inputs.get('admission_source_desc', 'Emergency Room'))
    input_data['diag_1_category'] = str(user_inputs.get('diag_1_category', 'Circulatory'))
    input_data['diag_2_category'] = str(user_inputs.get('diag_2_category', 'Diabetes'))
    input_data['diag_3_category'] = str(user_inputs.get('diag_3_category', 'Other'))

    # Construct single-row DataFrame with strict column order
    df = pd.DataFrame([input_data])
    missing_cols = [c for c in EXPECTED_INPUT_COLUMNS if c not in df.columns]
    if missing_cols:
        raise InputValidationError(f"Constructed input vector missing expected columns: {missing_cols}")

    return df[EXPECTED_INPUT_COLUMNS]


def predict_readmission_risk(
    model: Pipeline,
    input_df: pd.DataFrame,
    decision_threshold: float = 0.50
) -> Dict[str, Any]:
    """Execute model prediction and format results with calibrated disclaimers.

    Parameters
    ----------
    model : Pipeline
        Trained Scikit-learn Pipeline.
    input_df : pd.DataFrame
        Single-row feature DataFrame matching EXPECTED_INPUT_COLUMNS.
    decision_threshold : float, default=0.50
        Classification decision boundary applied to the uncalibrated probability estimate.

    Returns
    -------
    Dict[str, Any]
        Dictionary containing estimated probability, binary decision, risk tier,
        calibration status, and top empirical risk drivers.
    """
    if len(input_df) != 1:
        raise InputValidationError(f"Expected single-row input DataFrame, got shape {input_df.shape}.")

    try:
        prob = float(model.predict_proba(input_df)[0][1])
    except Exception as e:
        raise PredictionError(f"Model prediction failed: {str(e)}") from e

    predicted_class = int(prob >= decision_threshold)

    # Risk tiers are analytical heuristic bands, not clinically validated cutoffs
    if prob >= 0.50:
        tier_label = "Elevated Model Score (>= 50%)"
        tier_level = "high"
        tier_description = (
            "Model estimate meets or exceeds the 0.50 classification threshold. "
            "In the held-out test cohort, encounters flagged by this threshold captured 52.3% of "
            "actual 30-day readmissions, with a precision of 16.5% (~1 in 6 true readmissions)."
        )
    elif prob >= 0.30:
        tier_label = "Intermediate Model Score (30% – 49%)"
        tier_level = "moderate"
        tier_description = (
            "Model estimate is above population prevalence (10.9% test base rate) but below the "
            "binary decision threshold. Indicates moderate multi-morbidity or prior utilization."
        )
    else:
        tier_label = "Baseline Model Score (< 30%)"
        tier_level = "low"
        tier_description = (
            "Model estimate is consistent with low historical hospital utilization and "
            "lower observed readmission frequencies."
        )

    # Identify primary observed risk drivers from inputs
    row = input_df.iloc[0]
    risk_factors: List[str] = []

    num_inpatient = row.get('number_inpatient', 0)
    if num_inpatient >= 2:
        risk_factors.append(
            f"Prior Inpatient Admissions ({int(num_inpatient)} visits): "
            "Strongest positive regression coefficient in model (beta = +0.3517, p < 1e-15)."
        )
    elif num_inpatient == 1:
        risk_factors.append(
            "Prior Inpatient Admissions (1 visit): Associated with elevated readmission risk "
            "(12.9% observed rate vs 8.4% for 0 prior visits)."
        )

    num_diagnoses = row.get('number_diagnoses', 0)
    if num_diagnoses >= 8:
        risk_factors.append(
            f"Multi-Morbidity ({int(num_diagnoses)} recorded diagnoses): "
            "Positive regression coefficient (beta = +0.0933) reflecting compounding chronic burden."
        )

    age_midpoint = row.get('age_midpoint', 65)
    if age_midpoint >= 75:
        risk_factors.append(
            f"Age Midpoint ({int(age_midpoint)} years): "
            "Positive regression coefficient (beta = +0.0777) reflecting geriatric vulnerability."
        )

    if row.get('change') == 'Ch':
        risk_factors.append(
            "Inpatient Medication Titration (change = 'Ch'): "
            "Statistically significant association with readmission (chi2 = 215.94, p < 1e-46), "
            "reflecting acute glycemic volatility."
        )

    if row.get('number_emergency', 0) >= 1:
        risk_factors.append(
            f"Emergency Utilization ({int(row['number_emergency'])} visits): "
            "Indicator of acute unscheduled healthcare utilization in previous 12 months."
        )

    if not risk_factors:
        risk_factors.append(
            "Encounter profile shows zero prior inpatient visits and lower multi-morbidity indices."
        )

    return {
        'probability': prob,
        'probability_percent': round(prob * 100, 2),
        'predicted_class': predicted_class,
        'decision_threshold': decision_threshold,
        'risk_tier': tier_label,
        'tier_level': tier_level,
        'tier_description': tier_description,
        'is_calibrated': False,
        'calibration_note': (
            "Caution: This probability is an uncalibrated logistic regression output. "
            "It reflects relative statistical risk ranking within this dataset, not a direct "
            "calibrated clinical likelihood of real-world patient readmission."
        ),
        'risk_factors': risk_factors
    }
