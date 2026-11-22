import numpy as np
import pytest
from scipy.stats import spearmanr
from sklearn.ensemble import IsolationForest as SklearnIsolationForest
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import LocalOutlierFactor as SklearnLocalOutlierFactor

from netmine.iforest import IsolationForest, IsolationForestError, average_path_length
from netmine.lof import LocalOutlierFactor, LofError


def sample_data(seed=0, n_reference=1500, n_query=400, n_outliers=60, dimensions=8):
    rng = np.random.default_rng(seed)
    reference = rng.normal(size=(n_reference, dimensions))
    inliers = rng.normal(size=(n_query, dimensions))
    outliers = rng.normal(size=(n_outliers, dimensions)) * 3.5 + 4.0
    query = np.vstack([inliers, outliers])
    truth = np.r_[np.zeros(n_query, dtype=int), np.ones(n_outliers, dtype=int)]
    return reference, query, truth


def test_lof_matches_the_library_on_new_points():
    reference, query, _ = sample_data()
    mine = LocalOutlierFactor(n_neighbors=20, block_size=128).fit(reference)
    theirs = SklearnLocalOutlierFactor(n_neighbors=20, novelty=True).fit(reference)
    difference = np.abs(mine.score(query) - (-theirs.score_samples(query)))
    assert difference.max() < 1e-8


def test_lof_matches_the_library_on_fitted_points():
    reference, _, _ = sample_data()
    mine = LocalOutlierFactor(n_neighbors=15, block_size=256).fit(reference)
    theirs = SklearnLocalOutlierFactor(n_neighbors=15).fit(reference)
    difference = np.abs(mine.score_fitted() - (-theirs.negative_outlier_factor_))
    assert difference.max() < 1e-8


def test_lof_block_size_does_not_change_the_result():
    reference, query, _ = sample_data()
    small = LocalOutlierFactor(n_neighbors=10, block_size=17).fit(reference).score(query)
    large = LocalOutlierFactor(n_neighbors=10, block_size=4096).fit(reference).score(query)
    assert np.abs(small - large).max() < 1e-10


def test_lof_separates_planted_outliers():
    reference, query, truth = sample_data()
    scores = LocalOutlierFactor(n_neighbors=20, block_size=128).fit(reference).score(query)
    assert roc_auc_score(truth, scores) > 0.95


def test_lof_rejects_bad_input():
    with pytest.raises(LofError):
        LocalOutlierFactor(n_neighbors=0)
    with pytest.raises(LofError):
        LocalOutlierFactor(n_neighbors=5).score(np.zeros((3, 2)))
    with pytest.raises(LofError):
        LocalOutlierFactor(n_neighbors=5).fit(np.zeros((4, 2)))
    with pytest.raises(LofError):
        LocalOutlierFactor(n_neighbors=2).fit(np.zeros((10, 3))).score(np.zeros((5, 2)))


def test_average_path_length_known_values():
    values = average_path_length(np.array([0.0, 1.0, 2.0, 10.0]))
    assert values[0] == 0.0
    assert values[1] == 0.0
    assert values[2] == 1.0
    assert abs(values[3] - 3.748880) < 1e-5


def test_iforest_agrees_with_the_library_ranking():
    reference, query, _ = sample_data(seed=3)
    mine = IsolationForest(n_trees=200, subsample_size=256, random_seed=11).fit(reference).score(query)
    theirs = -SklearnIsolationForest(n_estimators=200, max_samples=256, random_state=11).fit(reference).score_samples(query)
    assert spearmanr(mine, theirs).statistic > 0.9


def test_iforest_separates_planted_outliers():
    reference, query, truth = sample_data(seed=5)
    scores = IsolationForest(n_trees=200, subsample_size=256, random_seed=5).fit(reference).score(query)
    assert roc_auc_score(truth, scores) > 0.95


def test_iforest_scores_stay_in_the_unit_interval():
    reference, query, _ = sample_data(seed=7)
    scores = IsolationForest(n_trees=50, subsample_size=64, random_seed=7).fit(reference).score(query)
    assert scores.min() > 0.0
    assert scores.max() < 1.0


def test_iforest_is_deterministic_for_a_fixed_seed():
    reference, query, _ = sample_data(seed=9)
    first = IsolationForest(n_trees=40, subsample_size=64, random_seed=13).fit(reference).score(query)
    second = IsolationForest(n_trees=40, subsample_size=64, random_seed=13).fit(reference).score(query)
    assert np.array_equal(first, second)


def test_iforest_handles_constant_columns():
    rng = np.random.default_rng(2)
    reference = np.hstack([rng.normal(size=(500, 3)), np.zeros((500, 2))])
    scores = IsolationForest(n_trees=30, subsample_size=64, random_seed=2).fit(reference).score(reference)
    assert np.isfinite(scores).all()


def test_iforest_rejects_bad_input():
    with pytest.raises(IsolationForestError):
        IsolationForest(n_trees=0)
    with pytest.raises(IsolationForestError):
        IsolationForest(subsample_size=2)
    with pytest.raises(IsolationForestError):
        IsolationForest().score(np.zeros((5, 2)))
    with pytest.raises(IsolationForestError):
        IsolationForest().fit(np.zeros((3, 2)))
