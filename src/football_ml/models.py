"""Factory for the 5 models, each wrapped in an identical preprocessing
pipeline so comparisons between them are about the estimator only.
"""
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

from . import config
from .features import build_preprocessor
from .mlp import MLPClassifier


def get_models() -> dict:
    """Return {model_name: unfitted sklearn Pipeline} for all 5 models."""
    return {
        "LogisticRegression": Pipeline(
            [
                ("prep", build_preprocessor()),
                (
                    "clf",
                    LogisticRegression(
                        max_iter=2000,
                        class_weight="balanced",
                        random_state=config.RANDOM_STATE,
                    ),
                ),
            ]
        ),
        "RandomForest": Pipeline(
            [
                ("prep", build_preprocessor()),
                (
                    "clf",
                    RandomForestClassifier(
                        n_estimators=400,
                        max_depth=None,
                        min_samples_leaf=2,
                        class_weight="balanced",
                        random_state=config.RANDOM_STATE,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
        "LightGBM": Pipeline(
            [
                ("prep", build_preprocessor()),
                (
                    "clf",
                    LGBMClassifier(
                        n_estimators=400,
                        learning_rate=0.03,
                        num_leaves=31,
                        class_weight="balanced",
                        random_state=config.RANDOM_STATE,
                        verbose=-1,
                    ),
                ),
            ]
        ),
        "SVM": Pipeline(
            [
                ("prep", build_preprocessor()),
                (
                    "clf",
                    SVC(
                        kernel="rbf",
                        C=2.0,
                        gamma="scale",
                        class_weight="balanced",
                        probability=True,
                        random_state=config.RANDOM_STATE,
                    ),
                ),
            ]
        ),
        "MLP (PyTorch)": Pipeline(
            [
                ("prep", build_preprocessor()),
                ("clf", MLPClassifier(random_state=config.RANDOM_STATE)),
            ]
        ),
    }
