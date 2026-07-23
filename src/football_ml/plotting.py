"""Matplotlib plots styled with the project's validated categorical palette
(see dataviz reference palette: blue/orange/aqua/yellow/magenta, CVD-checked
adjacent-pair order) rather than matplotlib's default cycle.
"""
import matplotlib.pyplot as plt
import numpy as np

from . import config

PALETTE = config.PALETTE


def _style_axes(ax):
    ax.set_facecolor(PALETTE["surface"])
    ax.figure.set_facecolor(PALETTE["surface"])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(PALETTE["grid"])
    ax.spines["bottom"].set_color(PALETTE["muted"])
    ax.tick_params(colors=PALETTE["ink_secondary"])
    ax.xaxis.label.set_color(PALETTE["ink"])
    ax.yaxis.label.set_color(PALETTE["ink"])
    ax.title.set_color(PALETTE["ink"])
    ax.grid(axis="y", color=PALETTE["grid"], linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)


def plot_model_comparison(results_df, metric_left="test_accuracy", metric_right="test_f1_macro", save_path=None):
    """Grouped bar chart comparing all 5 models on two holdout metrics.

    Two related metrics of the same 0-1 scale are shown as grouped bars on
    one shared axis (never a dual-axis chart).
    """
    models = results_df.index.tolist()
    x = np.arange(len(models))
    width = 0.35

    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.bar(
        x - width / 2, results_df[metric_left], width,
        label="Accuracy", color=PALETTE["series"][0],
        edgecolor=PALETTE["surface"], linewidth=1,
    )
    ax.bar(
        x + width / 2, results_df[metric_right], width,
        label="Macro F1", color=PALETTE["series"][1],
        edgecolor=PALETTE["surface"], linewidth=1,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=15, ha="right")
    ax.set_ylabel("Score (holdout test set)")
    ax.set_ylim(0, 1)
    ax.set_title("Model comparison — chronological holdout test set")
    legend = ax.legend(frameon=False, labelcolor=PALETTE["ink"])
    _style_axes(ax)

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, facecolor=fig.get_facecolor())
    return fig


def plot_confusion_matrix(cm, model_name, save_path=None):
    """Single-hue sequential heatmap (blue ramp) for one model's confusion matrix."""
    cm = np.asarray(cm)
    labels = config.TARGET_CLASSES
    cmap = plt.cm.colors.LinearSegmentedColormap.from_list(
        "seq_blue", PALETTE["sequential_blue"]
    )

    fig, ax = plt.subplots(figsize=(4.5, 4))
    im = ax.imshow(cm, cmap=cmap)
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"{model_name}")

    thresh = cm.max() / 2
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            color = PALETTE["surface"] if cm[i, j] > thresh else PALETTE["ink"]
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", color=color)

    ax.set_facecolor(PALETTE["surface"])
    fig.patch.set_facecolor(PALETTE["surface"])
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, facecolor=fig.get_facecolor())
    return fig
