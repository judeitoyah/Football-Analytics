"""Metrics, confusion matrices, and TimeSeriesSplit cross-validation."""
import json

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    log_loss,
)
from sklearn.model_selection import TimeSeriesSplit
from sklearn.base import clone

from . import config


def evaluate_holdout(model, X_test, y_test) -> dict:
    """Fitted-model metrics on the chronological holdout test set."""
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)
    # All 5 models are wrapped as Pipeline(..., ("clf", <estimator>)), so the
    # fitted classifier's own classes_ gives the correct predict_proba column order.
    classes = list(model.named_steps["clf"].classes_)

    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "f1_macro": f1_score(y_test, y_pred, average="macro"),
        "log_loss": log_loss(y_test, y_proba, labels=classes),
        "classification_report": classification_report(
            y_test, y_pred, output_dict=True, zero_division=0
        ),
        "confusion_matrix": confusion_matrix(
            y_test, y_pred, labels=config.TARGET_CLASSES
        ).tolist(),
    }


def cross_validate_time_series(model_builder, X_train, y_train, n_splits=config.CV_SPLITS) -> dict:
    """TimeSeriesSplit expanding-window CV on the training slice only.

    `model_builder` is a zero-arg callable returning a fresh unfitted
    pipeline (so each fold trains from scratch, not by mutating a fit model).
    """
    tscv = TimeSeriesSplit(n_splits=n_splits)
    X_train = X_train.reset_index(drop=True)
    y_train = y_train.reset_index(drop=True)

    fold_acc, fold_f1 = [], []
    for train_idx, val_idx in tscv.split(X_train):
        model = clone(model_builder())
        model.fit(X_train.iloc[train_idx], y_train.iloc[train_idx])
        y_pred = model.predict(X_train.iloc[val_idx])
        fold_acc.append(accuracy_score(y_train.iloc[val_idx], y_pred))
        fold_f1.append(f1_score(y_train.iloc[val_idx], y_pred, average="macro"))

    return {
        "cv_accuracy_mean": float(np.mean(fold_acc)),
        "cv_accuracy_std": float(np.std(fold_acc)),
        "cv_f1_macro_mean": float(np.mean(fold_f1)),
        "cv_f1_macro_std": float(np.std(fold_f1)),
        "cv_accuracy_folds": fold_acc,
        "cv_f1_macro_folds": fold_f1,
    }


def save_results(results: dict, path):
    with open(path, "w") as f:
        json.dump(results, f, indent=2, default=str)


def results_to_table(results: dict) -> pd.DataFrame:
    rows = []
    for name, r in results.items():
        rows.append(
            {
                "model": name,
                "cv_accuracy": r["cv"]["cv_accuracy_mean"],
                "cv_f1_macro": r["cv"]["cv_f1_macro_mean"],
                "test_accuracy": r["holdout"]["accuracy"],
                "test_f1_macro": r["holdout"]["f1_macro"],
                "test_log_loss": r["holdout"]["log_loss"],
            }
        )
    return pd.DataFrame(rows).set_index("model")
