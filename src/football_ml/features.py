"""Shared preprocessing: scale numeric columns, one-hot encode categoricals.

One ColumnTransformer is reused across all 5 models so every model sees the
exact same feature space — the only thing that differs between them is the
estimator itself.
"""
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from . import config


def build_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), config.NUMERIC_COLS),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                config.CATEGORICAL_COLS,
            ),
        ]
    )
