import numpy as np
import pytest

from netmine.evaluate import (
    EvaluationError,
    detection_report,
    precision_recall_at_k,
    ranking_metrics,
    threshold_at_fpr,
)
from netmine.scoring import rank_normalise


def test_precision_and_recall_at_k_on_a_known_ranking():
    scores = np.array([0.9, 0.8, 0.7, 0.6, 0.5])
    labels = np.array([1, 0, 1, 0, 0])
    precision, recall = precision_recall_at_k(scores, labels, 2)
    assert precision == 0.5
    assert recall == 0.5
    precision, recall = precision_recall_at_k(scores, labels, 3)
    assert abs(precision - 2 / 3) < 1e-12
    assert recall == 1.0


def test_k_larger_than_the_sample_is_clamped():
    scores = np.array([0.9, 0.1, 0.5])
    labels = np.array([1, 0, 0])
    precision, recall = precision_recall_at_k(scores, labels, 100)
    assert abs(precision - 1 / 3) < 1e-12
    assert recall == 1.0


def test_ranking_metrics_reports_the_base_rate():
    rng = np.random.default_rng(0)
    scores = np.r_[rng.normal(size=800), rng.normal(3.0, 1.0, size=200)]
    labels = np.r_[np.zeros(800, dtype=int), np.ones(200, dtype=int)]
    report = ranking_metrics(scores, labels, [50, 100])
    assert abs(report["base_rate"] - 0.2) < 1e-12
    assert report["n"] == 1000
    assert report["roc_auc"] > 0.9
    assert "precision_at_50" in report and "recall_at_100" in report


def test_single_class_labels_are_rejected():
    with pytest.raises(EvaluationError):
        ranking_metrics(np.array([1.0, 2.0]), np.array([0, 0]), [1])
    with pytest.raises(EvaluationError):
        ranking_metrics(np.array([1.0, 2.0]), np.array([1, 1]), [1])


def test_mismatched_lengths_are_rejected():
    with pytest.raises(EvaluationError):
        ranking_metrics(np.array([1.0, 2.0]), np.array([0, 1, 1]), [1])


def test_threshold_delivers_the_requested_false_positive_rate():
    rng = np.random.default_rng(1)
    scores = np.r_[rng.normal(size=5000), rng.normal(4.0, 1.0, size=500)]
    labels = np.r_[np.zeros(5000, dtype=int), np.ones(500, dtype=int)]
    for target in [0.01, 0.05, 0.1]:
        threshold = threshold_at_fpr(scores, labels, target)
        observed = (scores[labels == 0] >= threshold).mean()
        assert abs(observed - target) < 0.005


def test_detection_report_covers_every_attack_category():
    rng = np.random.default_rng(2)
    scores = np.r_[rng.normal(size=900), rng.normal(3.0, 1.0, size=60), rng.normal(1.0, 1.0, size=40)]
    labels = np.r_[np.zeros(900, dtype=int), np.ones(100, dtype=int)]
    categories = np.array(["Normal"] * 900 + ["Loud"] * 60 + ["Quiet"] * 40)
    rows = detection_report(scores, labels, categories, [0.05], "Normal")
    assert len(rows) == 1
    per_category = rows[0]["per_category"]
    assert set(per_category) == {"Loud", "Quiet"}
    assert per_category["Loud"]["support"] == 60
    assert per_category["Loud"]["recall"] > per_category["Quiet"]["recall"]


def test_invalid_false_positive_rate_is_rejected():
    scores = np.array([1.0, 2.0, 3.0])
    labels = np.array([0, 0, 1])
    with pytest.raises(EvaluationError):
        threshold_at_fpr(scores, labels, 0.0)
    with pytest.raises(EvaluationError):
        threshold_at_fpr(scores, labels, 1.0)


def test_rank_normalise_spans_the_unit_interval():
    values = np.array([5.0, 1.0, 3.0, 9.0])
    ranks = rank_normalise(values)
    assert ranks.min() == 0.0
    assert ranks.max() == 1.0
    assert np.argsort(ranks).tolist() == np.argsort(values).tolist()


def test_rank_normalise_gives_ties_the_same_value():
    values = np.array([2.0, 2.0, 2.0, 5.0])
    ranks = rank_normalise(values)
    assert ranks[0] == ranks[1] == ranks[2]
    assert ranks[3] > ranks[0]


def test_rank_normalise_is_invariant_to_monotone_rescaling():
    rng = np.random.default_rng(3)
    values = rng.normal(size=200)
    assert np.allclose(rank_normalise(values), rank_normalise(np.exp(values)))
