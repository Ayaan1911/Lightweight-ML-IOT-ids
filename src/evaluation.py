"""Metrics from predictions; test-set scoring of frozen models (logged)."""
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, f1_score
from .common import *


def compute(y, p, n_classes):
    """All metrics from labels/predictions. Single source of truth (verify.py reuses it)."""
    labels = list(range(n_classes))
    cm = confusion_matrix(y, p, labels=labels)
    pr, rc, f1, _ = precision_recall_fscore_support(y, p, labels=labels, average=None, zero_division=0)
    tp = np.diag(cm); fp = cm.sum(0) - tp; fn = cm.sum(1) - tp; tn = cm.sum() - tp - fp - fn
    fpr_c = fp / np.maximum(fp + tn, 1)
    return {
        "macro_f1": float(f1.mean()), "accuracy": float(tp.sum() / cm.sum()),
        "precision": float(pr.mean()), "recall": float(rc.mean()), "fpr": float(fpr_c.mean()) if n_classes > 2 else float(fpr_c[1]),
        "per_class_f1": f1.tolist(), "cm": cm.tolist(),
    }


def val_f1(y, p):
    return float(f1_score(y, p, average="macro", zero_division=0))


def score_test(model_id, task, y_test, pred, extra=None):
    """Score a frozen model on test ONCE; logs access, saves predictions, cm csv/png, metrics json."""
    n = 2 if task == "binary" else len(CLASSES)
    log_test_access(model_id)
    np.savez(RES / f"pred_{model_id}.npz", y_true=y_test, y_pred=pred)
    m = compute(y_test, pred, n)
    names = ["Benign", "Attack"] if task == "binary" else CLASSES
    cm = pd.DataFrame(m["cm"], index=names, columns=names)
    cm.to_csv(RES / f"cm_{model_id}.csv")
    fig, ax = plt.subplots(figsize=(4 + n * 0.4, 4 + n * 0.4))
    ax.imshow(np.log1p(cm.values), cmap="Blues")
    for i in range(n):
        for j in range(n):
            ax.text(j, i, cm.values[i, j], ha="center", va="center", fontsize=7)
    ax.set_xticks(range(n)); ax.set_xticklabels(names, rotation=60, ha="right", fontsize=7)
    ax.set_yticks(range(n)); ax.set_yticklabels(names, fontsize=7)
    ax.set_xlabel("predicted"); ax.set_ylabel("true"); ax.set_title(model_id, fontsize=8)
    fig.tight_layout(); fig.savefig(FIG / f"cm_{model_id}.png", dpi=110); plt.close(fig)
    row = {"model_id": model_id, "task": task, **{k: v for k, v in m.items() if k not in ("cm", "per_class_f1")}, **(extra or {})}
    if task == "multi":
        row.update({f"f1_{c}": f for c, f in zip(CLASSES, m["per_class_f1"])})
    jdump(row, RES / f"metrics_{model_id}.json")
    return row
