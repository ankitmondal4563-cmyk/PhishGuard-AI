"""Metric helpers shared by the URL and text training scripts."""
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score)


def evaluate(y_true, proba, threshold: float = 0.5) -> dict:
    y_true = np.asarray(y_true).astype(int)
    proba = np.asarray(proba, dtype=float)
    pred = (proba >= threshold).astype(int)
    cm = confusion_matrix(y_true, pred, labels=[0, 1])
    tn, fp, fn, tp = (int(v) for v in cm.ravel())
    auc = None
    if len(np.unique(y_true)) == 2:
        auc = round(float(roc_auc_score(y_true, proba)), 4)
    return {
        "n": int(len(y_true)),
        "accuracy": round(float(accuracy_score(y_true, pred)), 4),
        "precision": round(float(precision_score(y_true, pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, pred, zero_division=0)), 4),
        "f1": round(float(f1_score(y_true, pred, zero_division=0)), 4),
        "roc_auc": auc,
        "false_positive_rate": round(fp / (fp + tn), 4) if (fp + tn) else None,
        "confusion_matrix": [[tn, fp], [fn, tp]],
    }


def save_confusion_matrix(cm, path, title: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(4, 3.6))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1], ["Legit", "Phishing"])
    ax.set_yticks([0, 1], ["Legit", "Phishing"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(title)
    top = max(max(r) for r in cm) or 1
    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i][j], ha="center", va="center",
                    color="white" if cm[i][j] > top / 2 else "black")
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)


def save_json(obj, path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
