"""Unit test suite for SWYNEX modeling and evaluation module (src/modeling.py)."""
import os
import tempfile
import numpy as np
import pandas as pd
import pytest
from sklearn.dummy import DummyClassifier
from src.modeling import (
    build_logistic_regression_pipeline,
    build_preprocessing_pipeline,
    create_patient_grouped_split,
    evaluate_model,
    map_binary_target,
    prepare_feature_matrix,
    save_metrics_to_json
)


@pytest.fixture
def sample_clinical_df():
    """Create a controlled synthetic clinical dataset for testing modeling functions."""
    return pd.DataFrame({
        'encounter_id': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        'patient_nbr': [101, 102, 101, 103, 104, 105, 102, 106, 107, 108],
        'time_in_hospital': [2, 5, 3, 7, 1, 4, 6, 2, 8, 3],
        'num_medications': [10, 15, 12, 20, 8, 14, 18, 9, 22, 11],
        'number_inpatient': [0, 1, 0, 2, 0, 0, 1, 0, 3, 0],
        'race': ['Caucasian', 'AfricanAmerican', 'Caucasian', 'Other', 'Caucasian', 'Asian', 'AfricanAmerican', 'Caucasian', 'Other', 'Caucasian'],
        'gender': ['Male', 'Female', 'Male', 'Female', 'Male', 'Female', 'Male', 'Female', 'Male', 'Female'],
        'admission_type_desc': ['Emergency', 'Elective', 'Emergency', 'Urgent', 'Emergency', 'Elective', 'Emergency', 'Urgent', 'Emergency', 'Elective'],
        'discharge_disposition_desc': ['Discharged to home', 'Discharged to home', 'Discharged to home', 'SNF', 'Discharged to home', 'Rehab', 'Discharged to home', 'SNF', 'Expired', 'Discharged to home'],
        'discharge_disposition_id': [1, 1, 1, 3, 1, 22, 1, 3, 11, 1],
        'readmitted': ['NO', '>30', 'NO', '<30', 'NO', '>30', '<30', 'NO', '<30', 'NO']
    })


def test_map_binary_target(sample_clinical_df):
    y = map_binary_target(sample_clinical_df, target_col='readmitted')
    assert len(y) == 10
    # Expected: row 3, 6, 8 are '<30' -> index 3, 6, 8 have value 1
    assert y.iloc[3] == 1
    assert y.iloc[6] == 1
    assert y.iloc[8] == 1
    assert y.iloc[0] == 0  # 'NO'
    assert y.iloc[1] == 0  # '>30'
    assert y.sum() == 3


def test_prepare_feature_matrix_leakage_exclusion(sample_clinical_df):
    X, y, groups, num_cols, cat_cols = prepare_feature_matrix(sample_clinical_df, exclude_leakage=True)
    assert 'encounter_id' not in X.columns
    assert 'patient_nbr' not in X.columns
    assert 'readmitted' not in X.columns
    assert 'discharge_disposition_desc' not in X.columns
    assert 'discharge_disposition_id' not in X.columns
    assert len(num_cols) == 3  # time_in_hospital, num_medications, number_inpatient
    assert len(cat_cols) == 3  # race, gender, admission_type_desc
    assert len(groups) == 10
    assert len(y) == 10


def test_create_patient_grouped_split(sample_clinical_df):
    X, y, groups, num_cols, cat_cols = prepare_feature_matrix(sample_clinical_df)
    X_train, X_test, y_train, y_test = create_patient_grouped_split(X, y, groups, test_size=0.3, random_state=42)

    # Verify no patient overlap between train and test
    train_patients = set(groups.iloc[X_train.index])
    test_patients = set(groups.iloc[X_test.index])
    overlap = train_patients.intersection(test_patients)
    assert len(overlap) == 0
    assert len(X_train) + len(X_test) == 10


def test_build_logistic_regression_pipeline(sample_clinical_df):
    X, y, groups, num_cols, cat_cols = prepare_feature_matrix(sample_clinical_df)
    pipe = build_logistic_regression_pipeline(num_cols, cat_cols, random_state=42)
    assert 'prep' in pipe.named_steps
    assert 'clf' in pipe.named_steps

    # Test fitting on sample data
    pipe.fit(X, y)
    preds = pipe.predict(X)
    assert len(preds) == len(y)


def test_evaluate_model(sample_clinical_df):
    X, y, groups, num_cols, cat_cols = prepare_feature_matrix(sample_clinical_df)
    dummy = DummyClassifier(strategy='most_frequent')
    dummy.fit(X, y)
    metrics = evaluate_model(dummy, X, y)

    assert 'accuracy' in metrics
    assert 'precision' in metrics
    assert 'recall' in metrics
    assert 'f1_score' in metrics
    assert 'roc_auc' in metrics
    assert 'pr_auc' in metrics
    assert 'confusion_matrix' in metrics
    assert metrics['total_test_samples'] == 10
    assert metrics['positive_support'] == 3
    assert metrics['negative_support'] == 7


def test_save_metrics_to_json():
    sample_metrics = {'model': 'LogisticRegression', 'accuracy': 0.6586, 'recall': 0.5226}
    with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as tmp:
        tmp_path = tmp.name

    try:
        save_metrics_to_json(sample_metrics, tmp_path)
        assert os.path.exists(tmp_path)
        with open(tmp_path, 'r', encoding='utf-8') as f:
            import json
            data = json.load(f)
        assert data['accuracy'] == 0.6586
        assert data['model'] == 'LogisticRegression'
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
