"""SWYNEX Clinical Data Science & Prediction Package."""

from src.utils import convert_age_to_midpoint, map_icd9_category, normalize_lab_tests, parse_ids_mapping
from src.modeling import (
    build_logistic_regression_pipeline,
    create_patient_grouped_split,
    evaluate_model,
    map_binary_target,
    prepare_feature_matrix,
    save_metrics_to_json
)
from src.prediction import (
    InputValidationError,
    ModelCompatibilityError,
    ModelNotFoundError,
    PredictionError,
    SYNTHETIC_PROFILES,
    load_model,
    predict_readmission_risk,
    validate_and_build_input_vector
)

__all__ = [
    'convert_age_to_midpoint',
    'map_icd9_category',
    'normalize_lab_tests',
    'parse_ids_mapping',
    'build_logistic_regression_pipeline',
    'create_patient_grouped_split',
    'evaluate_model',
    'map_binary_target',
    'prepare_feature_matrix',
    'save_metrics_to_json',
    'load_model',
    'validate_and_build_input_vector',
    'predict_readmission_risk',
    'SYNTHETIC_PROFILES',
    'PredictionError',
    'ModelNotFoundError',
    'ModelCompatibilityError',
    'InputValidationError'
]
