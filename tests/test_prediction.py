"""Unit tests for src/prediction.py.

Verifies schema validation, model artifact loading, input constraints,
probability bounds, uncalibrated probability disclaimers, and synthetic profiles.
"""
import os
import pytest
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.dummy import DummyClassifier

from src.prediction import (
    load_model,
    validate_and_build_input_vector,
    predict_readmission_risk,
    SYNTHETIC_PROFILES,
    ModelNotFoundError,
    ModelCompatibilityError,
    InputValidationError,
    EXPECTED_INPUT_COLUMNS,
    NUMERICAL_FEATURE_RANGES
)


@pytest.fixture(scope="module")
def trained_model():
    """Load the trained model artifact for testing."""
    model_path = os.path.join('reports', 'modeling', 'model_pipeline.joblib')
    if not os.path.exists(model_path):
        pytest.skip(f"Model artifact not found at {model_path}")
    return load_model(model_path)


def test_load_model_valid(trained_model):
    """Test that a valid model loads as a scikit-learn Pipeline with required steps."""
    assert isinstance(trained_model, Pipeline)
    assert 'prep' in trained_model.named_steps
    assert 'clf' in trained_model.named_steps


def test_load_model_missing_file_raises_error():
    """Test that loading a non-existent model file raises ModelNotFoundError."""
    with pytest.raises(ModelNotFoundError):
        load_model('non_existent_path_to_model.joblib')


def test_load_model_invalid_object_raises_compatibility_error(tmp_path):
    """Test that loading an object that is not a Pipeline raises ModelCompatibilityError."""
    import joblib
    fake_path = os.path.join(tmp_path, "not_a_pipeline.joblib")
    joblib.dump({"some": "dictionary"}, fake_path)

    with pytest.raises(ModelCompatibilityError):
        load_model(str(fake_path))


def test_validate_and_build_input_vector_valid():
    """Test that valid inputs construct a DataFrame matching EXPECTED_INPUT_COLUMNS."""
    valid_inputs = {
        'age_bracket': '[50-60)',
        'time_in_hospital': 4,
        'num_lab_procedures': 40,
        'num_procedures': 1,
        'num_medications': 12,
        'number_outpatient': 0,
        'number_emergency': 0,
        'number_inpatient': 1,
        'number_diagnoses': 5,
        'gender': 'Male',
        'race': 'Caucasian',
        'admission_type_desc': 'Emergency',
        'admission_source_desc': 'Emergency Room',
        'diag_1_category': 'Circulatory',
        'diag_2_category': 'Diabetes',
        'diag_3_category': 'Other',
        'max_glu_serum': 'Not Tested',
        'A1Cresult': 'Norm',
        'change': 'No',
        'diabetesMed': 'Yes',
        'insulin': 'Steady'
    }
    df = validate_and_build_input_vector(valid_inputs)
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 1
    assert list(df.columns) == EXPECTED_INPUT_COLUMNS
    assert df['age_midpoint'].iloc[0] == 55.0
    assert df['number_inpatient'].iloc[0] == 1.0


def test_validate_and_build_input_vector_invalid_age_bracket():
    """Test that an unrecognized age bracket raises InputValidationError."""
    inputs = {
        'age_bracket': '[120-130)',
        'time_in_hospital': 4,
        'num_lab_procedures': 40,
        'num_procedures': 1,
        'num_medications': 12,
        'number_outpatient': 0,
        'number_emergency': 0,
        'number_inpatient': 0,
        'number_diagnoses': 5
    }
    with pytest.raises(InputValidationError):
        validate_and_build_input_vector(inputs)


def test_validate_and_build_input_vector_out_of_range():
    """Test that out-of-range numerical values raise InputValidationError."""
    # time_in_hospital exceeds maximum of 14 days
    inputs = {
        'age_bracket': '[50-60)',
        'time_in_hospital': 45,  # Invalid
        'num_lab_procedures': 40,
        'num_procedures': 1,
        'num_medications': 12,
        'number_outpatient': 0,
        'number_emergency': 0,
        'number_inpatient': 0,
        'number_diagnoses': 5
    }
    with pytest.raises(InputValidationError, match="outside acceptable range"):
        validate_and_build_input_vector(inputs)


def test_validate_and_build_input_vector_missing_required_numerical():
    """Test that missing required numerical values raise InputValidationError."""
    inputs = {
        'age_bracket': '[50-60)',
        'time_in_hospital': 4,
        # missing num_lab_procedures
        'num_procedures': 1,
        'num_medications': 12,
        'number_outpatient': 0,
        'number_emergency': 0,
        'number_inpatient': 0,
        'number_diagnoses': 5
    }
    with pytest.raises(InputValidationError, match="Missing required numerical feature"):
        validate_and_build_input_vector(inputs)


def test_predict_readmission_risk_output(trained_model):
    """Test structure and types of prediction output dictionary."""
    valid_inputs = SYNTHETIC_PROFILES['synthetic_elevated_utilization']['inputs']
    df = validate_and_build_input_vector(valid_inputs)
    result = predict_readmission_risk(trained_model, df)

    assert isinstance(result, dict)
    assert 'probability' in result
    assert 'probability_percent' in result
    assert 'predicted_class' in result
    assert 'risk_tier' in result
    assert 'tier_level' in result
    assert 'is_calibrated' in result
    assert result['is_calibrated'] is False
    assert 0.0 <= result['probability'] <= 1.0
    assert result['predicted_class'] in [0, 1]
    assert len(result['risk_factors']) > 0


def test_all_synthetic_profiles_executable(trained_model):
    """Test that all synthetic profiles run through validation and prediction cleanly."""
    for key, profile in SYNTHETIC_PROFILES.items():
        assert 'inputs' in profile
        assert 'description' in profile
        df = validate_and_build_input_vector(profile['inputs'])
        result = predict_readmission_risk(trained_model, df)
        assert 0.0 <= result['probability'] <= 1.0
        assert result['tier_level'] in ['high', 'moderate', 'low']
