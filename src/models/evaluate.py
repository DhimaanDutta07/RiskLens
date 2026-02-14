import os
import json
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    roc_curve
)


def best_f1_threshold(y_true, y_prob) -> float:
    p, r, t = precision_recall_curve(y_true, y_prob)
    t = np.append(t, 1.0)
    f1 = (2 * p * r) / (p + r + 1e-12)
    i = int(np.nanargmax(f1))
    return float(t[i])


def _save_confusion_matrix(cm, out_path: str):
    plt.figure()
    plt.imshow(cm)
    plt.xticks([0, 1], ["0", "1"])
    plt.yticks([0, 1], ["0", "1"])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    for i in range(2):
        for j in range(2):
            plt.text(j, i, str(cm[i][j]), ha="center", va="center")
    plt.tight_layout()
    plt.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close()


def _save_pr_curve(y_true, y_prob, out_path: str):
    p, r, _ = precision_recall_curve(y_true, y_prob)
    plt.figure()
    plt.plot(r, p)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.tight_layout()
    plt.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close()


def _save_roc_curve(y_true, y_prob, out_path: str):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    plt.figure()
    plt.plot(fpr, tpr)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.tight_layout()
    plt.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close()


def _save_metrics_bar(roc_auc, pr_auc, out_path: str):
    plt.figure()
    plt.bar(["ROC_AUC", "PR_AUC"], [roc_auc, pr_auc])
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close()


def evaluate(y_true, y_prob, threshold: float | None = None) -> dict:
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)

    if threshold is None:
        threshold = best_f1_threshold(y_true, y_prob)

    y_pred = (y_prob >= threshold).astype(int)

    cm = confusion_matrix(y_true, y_pred)
    roc = float(roc_auc_score(y_true, y_prob))
    pr = float(average_precision_score(y_true, y_prob))

    return {
        "threshold": float(threshold),
        "roc_auc": roc,
        "pr_auc": pr,
        "confusion_matrix": cm.tolist(),
        "report": classification_report(y_true, y_pred, digits=4),
    }


def save_report(report: dict, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)


def save_graphs(y_true, y_prob, report: dict, out_dir: str = "artifacts/graphs", prefix: str = "model"):
    os.makedirs(out_dir, exist_ok=True)

    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)

    thr = float(report["threshold"])
    y_pred = (y_prob >= thr).astype(int)
    cm = confusion_matrix(y_true, y_pred).tolist()

    _save_confusion_matrix(cm, os.path.join(out_dir, f"{prefix}_confusion.png"))
    _save_pr_curve(y_true, y_prob, os.path.join(out_dir, f"{prefix}_pr_curve.png"))
    _save_roc_curve(y_true, y_prob, os.path.join(out_dir, f"{prefix}_roc_curve.png"))
    _save_metrics_bar(report["roc_auc"], report["pr_auc"], os.path.join(out_dir, f"{prefix}_metrics_bar.png"))
