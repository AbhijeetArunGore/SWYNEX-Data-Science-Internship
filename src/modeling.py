"""SWYNEX Modeling and Evaluation Utility Module (Task 3).

Provides modular, reproducible functions for clinical feature matrix preparation,
patient-grouped train-test splitting, scikit-learn preprocessing pipelines,
baseline model training, and clinical metric evaluation.
"""
from typing import Dict, List, Optional, Tuple, Union
import json
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score
)
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def map_binary_target(df: pd.DataFrame, target_col: str = 'readmitted') -> pd.Series:
    """Map the clinical readmission column to a binary target.

    Positive class (1): readmitted == '<30' (early readmission within 30 days)
    Negative class (0): readmitted is 'NO' or '>30'

    Parameters
    ----------
    df : pd.DataFrame
        Input clinical dataset.
    target_col : str, default='readmitted'
        Name of the readmission outcome column.

    Returns
    -------
    pd.Series
        Binary integer series (1 or 0).
    """
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in DataFrame.")
    return (df[target_col] == '<30').astype(int)


def prepare_feature_matrix(
    df: pd.DataFrame,
    target_col: str = 'readmitted',
    exclude_leakage: bool = True
) -> Tuple[pd.DataFrame, pd.Series, pd.Series, List[str], List[str]]:
    """Separate target from input features and drop identifiers and leakage columns.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned clinical dataset from Task 1.
    target_col : str, default='readmitted'
        Name of the readmission column.
    exclude_leakage : bool, default=True
        Whether to exclude discharge disposition to prevent terminal status leakage
        and post-discharge timing issues.

    Returns
    -------
    Tuple[pd.DataFrame, pd.Series, pd.Series, List[str], List[str]]
        (X, y, groups, numerical_cols, categorical_cols)
    """
    df_copy = df.copy()
    y = map_binary_target(df_copy, target_col=target_col)
    groups = df_copy['patient_nbr']

    # Standard drop list: Identifiers and redundant raw codes already standardized in Task 1
    drop_columns = [
        'encounter_id', 'patient_nbr', target_col,
        'admission_type_id', 'admission_source_id',
        'age', 'diag_1', 'diag_2', 'diag_3'
    ]

    # Exclude confirmed target leakage / post-discharge timing features
    if exclude_leakage:
        drop_columns.extend(['discharge_disposition_id', 'discharge_disposition_desc'])
    else:
        drop_columns.append('discharge_disposition_id')

    # Drop target column if created as temporary column
    if 'target' in df_copy.columns:
        drop_columns.append('target')

    # Retain only valid columns that exist in the DataFrame
    existing_drops = [c for c in drop_columns if c in df_copy.columns]
    X = df_copy.drop(columns=existing_drops)

    numerical_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()

    return X, y, groups, numerical_cols, categorical_cols


def create_patient_grouped_split(
    X: pd.DataFrame,
    y: pd.Series,
    groups: pd.Series,
    test_size: float = 0.2,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Split data into training and test sets grouped strictly by patient ID.

    Ensures that multiple encounters from the same patient never appear in both
    training and test partitions (eliminates patient-level evaluation leakage).

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix.
    y : pd.Series
        Target labels.
    groups : pd.Series
        Patient identifier series (e.g. patient_nbr).
    test_size : float, default=0.2
        Proportion of patient groups allocated to the test split.
    random_state : int, default=42
        Random seed for reproducibility.

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]
        (X_train, X_test, y_train, y_test)
    """
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_idx, test_idx = next(gss.split(X, y, groups=groups))

    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    # Assert 0 patient overlap
    train_patients = set(groups.iloc[train_idx])
    test_patients = set(groups.iloc[test_idx])
    overlap = train_patients.intersection(test_patients)
    if len(overlap) > 0:
        raise ValueError(f"Patient leakage detected: {len(overlap)} patients overlap between train and test!")

    return X_train, X_test, y_train, y_test


def build_preprocessing_pipeline(
    numerical_cols: List[str],
    categorical_cols: List[str]
) -> ColumnTransformer:
    """Construct a scikit-learn ColumnTransformer for preprocessing.

    Scales numerical variables with StandardScaler and encodes categorical
    variables with OneHotEncoder(handle_unknown='ignore').

    Parameters
    ----------
    numerical_cols : List[str]
        Continuous or discrete numerical column names.
    categorical_cols : List[str]
        Categorical feature names.

    Returns
    -------
    ColumnTransformer
        Fitted on training data only within a scikit-learn Pipeline.
    """
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_cols)
        ]
    )
    return preprocessor


def build_logistic_regression_pipeline(
    numerical_cols: List[str],
    categorical_cols: List[str],
    class_weight: Optional[str] = 'balanced',
    max_iter: int = 1000,
    random_state: int = 42
) -> Pipeline:
    """Build a complete scikit-learn Pipeline with preprocessing and Logistic Regression.

    Parameters
    ----------
    numerical_cols : List[str]
        List of numerical feature names.
    categorical_cols : List[str]
        List of categorical feature names.
    class_weight : Optional[str], default='balanced'
        Strategy to adjust weights inversely proportional to class frequencies.
    max_iter : int, default=1000
        Maximum iterations for solver convergence.
    random_state : int, default=42
        Random seed for solver reproducibility.

    Returns
    -------
    Pipeline
        Unfitted pipeline containing ('prep', ColumnTransformer) and ('clf', LogisticRegression).
    """
    preprocessor = build_preprocessing_pipeline(numerical_cols, categorical_cols)
    model = LogisticRegression(
        max_iter=max_iter,
        class_weight=class_weight,
        random_state=random_state
    )
    pipeline = Pipeline([
        ('prep', preprocessor),
        ('clf', model)
    ])
    return pipeline


def evaluate_model(
    model: Union[Pipeline, DummyClassifier],
    X_test: pd.DataFrame,
    y_test: pd.Series
) -> Dict[str, Union[float, int, List[List[int]]]]:
    """Evaluate a trained model on a held-out test partition.

    Calculates accuracy, precision, recall, F1-score, ROC-AUC, PR-AUC,
    and confusion matrix components.

    Parameters
    ----------
    model : Union[Pipeline, DummyClassifier]
        Fitted scikit-learn model or pipeline.
    X_test : pd.DataFrame
        Test features.
    y_test : pd.Series
        True test labels.

    Returns
    -------
    Dict[str, Union[float, int, List[List[int]]]]
        Dictionary of calculated evaluation metrics.
    """
    y_pred = model.predict(X_test)

    # Probabilities for positive class (1)
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)[:, 1]
    elif hasattr(model, "decision_function"):
        y_prob = model.decision_function(X_test)
    else:
        y_prob = y_pred

    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1])

    acc = float(accuracy_score(y_test, y_pred))
    prec = float(precision_score(y_test, y_pred, zero_division=0))
    rec = float(recall_score(y_test, y_pred, zero_division=0))
    f1 = float(f1_score(y_test, y_pred, zero_division=0))

    # ROC-AUC and PR-AUC require probabilities
    try:
        roc_auc = float(roc_auc_score(y_test, y_prob))
    except (ValueError, IndexError):
        roc_auc = 0.5

    try:
        pr_auc = float(average_precision_score(y_test, y_prob))
    except (ValueError, IndexError):
        pr_auc = float(y_test.mean())

    return {
        'total_test_samples': int(len(y_test)),
        'positive_support': int(y_test.sum()),
        'negative_support': int(len(y_test) - y_test.sum()),
        'true_negatives': tn,
        'false_positives': fp,
        'false_negatives': fn,
        'true_positives': tp,
        'accuracy': round(acc, 4),
        'precision': round(prec, 4),
        'recall': round(rec, 4),
        'f1_score': round(f1, 4),
        'roc_auc': round(roc_auc, 4),
        'pr_auc': round(pr_auc, 4),
        'confusion_matrix': cm.tolist()
    }


def save_metrics_to_json(metrics: dict, output_path: str) -> None:
    """Save calculated model metrics to a machine-readable JSON file.

    Parameters
    ----------
    metrics : dict
        Dictionary of evaluated model metrics.
    output_path : str
        Target file path.
    """
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=4)
