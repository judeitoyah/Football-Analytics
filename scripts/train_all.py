"""Train and evaluate all 5 models on the Football-Analytics dataset.

Usage:
    python scripts/train_all.py

Produces:
    results/metrics.json            - full CV + holdout metrics per model
    results/model_comparison.csv    - flat comparison table
    results/model_comparison.png    - grouped bar chart (accuracy / F1)
    results/confusion_matrix_*.png  - one heatmap per model
    results/shap_summary_*.png      - SHAP feature importance (tree models)
    models/*.joblib                 - fitted pipelines
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import joblib
import matplotlib.pyplot as plt
import shap

from football_ml import config, data, evaluate, models, plotting, shap_utils


def main():
    print("Loading data and recomputing the true 3-class target (H/D/A)...")
    df = data.build_dataset()
    X_train, X_test, y_train, y_test = data.chronological_split(df)
    print(f"Train: {len(X_train)} matches | Test (most recent {config.TEST_FRACTION:.0%}): {len(X_test)} matches")
    print(f"Train target distribution:\n{y_train.value_counts()}")

    model_builders = models.get_models()
    results = {}

    for name, builder in model_builders.items():
        print(f"\n=== {name} ===")
        t0 = time.time()

        print("  Running TimeSeriesSplit CV on the training slice...")
        cv_metrics = evaluate.cross_validate_time_series(
            lambda n=name: models.get_models()[n], X_train, y_train
        )
        print(f"  CV accuracy: {cv_metrics['cv_accuracy_mean']:.3f} +/- {cv_metrics['cv_accuracy_std']:.3f}")

        print("  Fitting on full training slice...")
        model = builder
        model.fit(X_train, y_train)

        holdout_metrics = evaluate.evaluate_holdout(model, X_test, y_test)
        print(f"  Holdout accuracy: {holdout_metrics['accuracy']:.3f} | macro F1: {holdout_metrics['f1_macro']:.3f}")

        results[name] = {"cv": cv_metrics, "holdout": holdout_metrics}

        safe_name = name.split(" ")[0]
        joblib.dump(model, config.MODELS_DIR / f"{safe_name}.joblib")

        fig = plotting.plot_confusion_matrix(
            holdout_metrics["confusion_matrix"], name,
            save_path=config.RESULTS_DIR / f"confusion_matrix_{safe_name}.png",
        )
        plt.close(fig)

        print(f"  Done in {time.time() - t0:.1f}s")

    print("\nSaving comparison table and chart...")
    evaluate.save_results(results, config.RESULTS_DIR / "metrics.json")
    table = evaluate.results_to_table(results)
    table.to_csv(config.RESULTS_DIR / "model_comparison.csv")
    print(table.round(3))

    fig = plotting.plot_model_comparison(table, save_path=config.RESULTS_DIR / "model_comparison.png")
    plt.close(fig)

    print("\nSHAP summary plots for tree-based models...")
    for name in ["RandomForest", "LightGBM"]:
        model = model_builders[name]
        sample = X_test.sample(min(300, len(X_test)), random_state=config.RANDOM_STATE)
        shap_values, feature_names, X_transformed = shap_utils.tree_shap_values(model, sample)

        fig = plt.figure(figsize=(8, 6))
        # Use the "H" (home win) class slice for a single, readable summary.
        class_idx = list(model.named_steps["clf"].classes_).index("H")
        shap.summary_plot(
            shap_values[class_idx], X_transformed, feature_names=feature_names,
            show=False, plot_size=None,
        )
        fig.tight_layout()
        fig.savefig(config.RESULTS_DIR / f"shap_summary_{name}.png", dpi=150)
        plt.close(fig)

    print(f"\nAll results written to {config.RESULTS_DIR}")
    print(f"All fitted models written to {config.MODELS_DIR}")


if __name__ == "__main__":
    main()
