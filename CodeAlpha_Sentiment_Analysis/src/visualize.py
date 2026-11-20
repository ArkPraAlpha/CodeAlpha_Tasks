# All charts (matplotlib only). Every function saves a PNG and returns its path.
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

COLORS = {"positive": "#2e8b57", "neutral": "#8c8c8c", "negative": "#c0392b"}


def _save(fig, out_dir, name):
    path = Path(out_dir) / name
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def plot_sentiment_distribution(df, out_dir, label_col="sentiment"):
    table = pd.crosstab(df["source"], df[label_col], normalize="index") * 100
    table = table.reindex(columns=["positive", "neutral", "negative"]).fillna(0)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    table.plot(kind="bar", stacked=True, ax=ax, color=[COLORS[c] for c in table.columns])
    ax.set_ylabel("% of texts")
    ax.set_xlabel("")
    ax.set_title("Sentiment distribution by source")
    ax.legend(title="", loc="upper right")
    plt.setp(ax.get_xticklabels(), rotation=0)
    return _save(fig, out_dir, "01_sentiment_by_source.png")


def plot_model_comparison(eval_df, out_dir):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = np.arange(len(eval_df))
    ax.bar(x - 0.2, eval_df["accuracy"], 0.4, label="Accuracy", color="#3b6ea5")
    ax.bar(x + 0.2, eval_df["f1_macro"], 0.4, label="Macro F1", color="#e08e45")
    ax.set_xticks(x)
    ax.set_xticklabels(eval_df["model"], rotation=15)
    ax.set_ylim(0, 1)
    ax.set_title("Model comparison (vs. ground-truth labels)")
    ax.legend()
    return _save(fig, out_dir, "02_model_comparison.png")


def plot_confusion(y_true, y_pred, out_dir, title="Confusion matrix (ensemble)"):
    labels = ["negative", "neutral", "positive"]
    cm = pd.crosstab(pd.Series(y_true, name="true"), pd.Series(y_pred, name="pred"))
    cm = cm.reindex(index=labels, columns=labels).fillna(0).astype(int)
    fig, ax = plt.subplots(figsize=(5, 4.5))
    ax.imshow(cm.values, cmap="Blues")
    ax.set_xticks(range(3)); ax.set_xticklabels(labels)
    ax.set_yticks(range(3)); ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True"); ax.set_title(title)
    for i in range(3):
        for j in range(3):
            ax.text(j, i, cm.values[i, j], ha="center", va="center",
                    color="white" if cm.values[i, j] > cm.values.max() / 2 else "black")
    return _save(fig, out_dir, "03_confusion_matrix.png")


def plot_emotions(emotion_by_sentiment, out_dir):
    data = emotion_by_sentiment.reindex(["positive", "neutral", "negative"]).dropna(how="all")
    fig, ax = plt.subplots(figsize=(9, 3.8))
    im = ax.imshow(data.values, cmap="YlOrRd", aspect="auto")
    ax.set_xticks(range(data.shape[1])); ax.set_xticklabels(data.columns, rotation=20)
    ax.set_yticks(range(data.shape[0])); ax.set_yticklabels(data.index)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            ax.text(j, i, f"{data.values[i, j]:.2f}", ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax, label="avg. emotion words per text")
    ax.set_title("Emotions by sentiment class")
    return _save(fig, out_dir, "04_emotions_heatmap.png")


def plot_aspects(aspect_summary, out_dir):
    d = aspect_summary.sort_values("mean_sentiment")
    colors = [COLORS["negative"] if v < -0.05 else COLORS["positive"] if v > 0.05 else COLORS["neutral"]
              for v in d["mean_sentiment"]]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.barh(d["aspect"], d["mean_sentiment"], color=colors)
    for y, (v, n) in enumerate(zip(d["mean_sentiment"], d["mentions"])):
        ax.text(v + (0.01 if v >= 0 else -0.01), y, f"n={n}", va="center",
                ha="left" if v >= 0 else "right", fontsize=8)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Mean sentiment of clauses mentioning the aspect")
    ax.set_title("Aspect-based sentiment")
    return _save(fig, out_dir, "05_aspect_sentiment.png")


def plot_trend(ts, out_dir, by_source=None):
    fig, ax = plt.subplots(figsize=(9, 4.5))
    if by_source:
        for name, t in by_source.items():
            ax.plot(t["period"], t["mean_score"], marker=".", alpha=0.5, label=name)
    ax.plot(ts["period"], ts["mean_score"], marker="o", color="black", linewidth=2.2, label="all sources")
    ax.plot(ts["period"], ts["rolling_mean"], linestyle="--", color="#3b6ea5", label="3-period average")
    flagged = ts[ts["anomaly"]]
    ax.scatter(flagged["period"], flagged["mean_score"], s=160, facecolors="none",
               edgecolors="red", linewidths=2, label="anomaly", zorder=5)
    ax.axhline(0, color="grey", linewidth=0.7)
    ax.set_ylabel("Mean sentiment score")
    ax.set_title("Sentiment trend over time")
    ax.legend(fontsize=8, ncol=2)
    fig.autofmt_xdate()
    return _save(fig, out_dir, "06_sentiment_trend.png")


def plot_top_terms(features, out_dir, n=10):
    classes = [c for c in ["negative", "neutral", "positive"] if c in set(features["class"])]
    fig, axes = plt.subplots(1, len(classes), figsize=(4.2 * len(classes), 4.5))
    axes = np.atleast_1d(axes)
    for ax, cls in zip(axes, classes):
        d = features[features["class"] == cls].head(n).iloc[::-1]
        ax.barh(d["term"], d["weight"], color=COLORS[cls])
        ax.set_title(f"Top terms: {cls}")
    return _save(fig, out_dir, "07_top_terms.png")
