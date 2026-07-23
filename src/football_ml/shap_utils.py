"""SHAP explainability for the tree-based models (RandomForest, LightGBM)."""
import numpy as np
import shap

from . import config


def get_feature_names(preprocessor) -> list:
    num_names = config.NUMERIC_COLS
    cat_encoder = preprocessor.named_transformers_["cat"]
    cat_names = list(cat_encoder.get_feature_names_out(config.CATEGORICAL_COLS))
    return num_names + cat_names


def tree_shap_values(pipeline, X_sample):
    """Return (shap_values, feature_names, X_transformed) for a fitted
    tree-based Pipeline (RandomForest or LightGBM step named 'clf').
    """
    preprocessor = pipeline.named_steps["prep"]
    clf = pipeline.named_steps["clf"]
    X_transformed = preprocessor.transform(X_sample)
    feature_names = get_feature_names(preprocessor)

    explainer = shap.TreeExplainer(clf)
    shap_values = explainer.shap_values(X_transformed)

    # Multi-class tree explainers return a list (one array per class) or a
    # 3-D array depending on the SHAP/library version; normalize to a list.
    if isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
        shap_values = [shap_values[:, :, i] for i in range(shap_values.shape[2])]

    return shap_values, feature_names, X_transformed
