import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


class EvaluationError(Exception):
    pass


def _check_inputs(scores: np.ndarray, labels: np.ndarray) -> None:
    if len(scores) != len(labels):
        raise EvaluationError("scores and labels must have the same length")
    if len(scores) == 0:
        raise EvaluationError("scores must not be empty")
    positives = int(labels.sum())
    if positives == 0 or positives == len(labels):
        raise EvaluationError("labels must contain both classes")


def precision_recall_at_k(scores: np.ndarray, labels: np.ndarray, k: int) -> tuple[float, float]:
    _check_inputs(scores, labels)
    if k < 1:
        raise EvaluationError("k must be at least 1")
    limit = min(k, len(scores))
    order = np.argsort(-scores, kind="mergesort")[:limit]
    hits = int(labels[order].sum())
    return hits / limit, hits / int(labels.sum())


def ranking_metrics(scores: np.ndarray, labels: np.ndarray, ks: list) -> dict:
    _check_inputs(scores, labels)
    report = {
        "roc_auc": float(roc_auc_score(labels, scores)),
        "pr_auc": float(average_precision_score(labels, scores)),
        "base_rate": float(labels.mean()),
        "n": int(len(labels)),
    }
    for k in ks:
        precision, recall = precision_recall_at_k(scores, labels, k)
        report["precision_at_" + str(k)] = precision
        report["recall_at_" + str(k)] = recall
    return report


def threshold_at_fpr(scores: np.ndarray, labels: np.ndarray, fpr: float) -> float:
    _check_inputs(scores, labels)
    if not 0.0 < fpr < 1.0:
        raise EvaluationError("fpr must be in (0, 1)")
    normal = scores[labels == 0]
    if normal.size == 0:
        raise EvaluationError("no normal flows available to set a threshold")
    return float(np.quantile(normal, 1.0 - fpr))


def detection_report(scores: np.ndarray, labels: np.ndarray, categories: np.ndarray, fprs: list, normal_category: str) -> list:
    _check_inputs(scores, labels)
    rows = []
    for fpr in fprs:
        threshold = threshold_at_fpr(scores, labels, fpr)
        flagged = scores >= threshold
        row = {
            "target_fpr": float(fpr),
            "threshold": threshold,
            "observed_fpr": float(flagged[labels == 0].mean()),
            "overall_recall": float(flagged[labels == 1].mean()),
            "per_category": {},
        }
        for category in sorted(set(categories.tolist())):
            if category == normal_category:
                continue
            mask = categories == category
            row["per_category"][category] = {
                "support": int(mask.sum()),
                "recall": float(flagged[mask].mean()) if mask.any() else 0.0,
            }
        rows.append(row)
    return rows
