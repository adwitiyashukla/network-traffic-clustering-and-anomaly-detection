import time

import numpy as np
from scipy.stats import spearmanr
from sklearn.ensemble import IsolationForest as SklearnIsolationForest
from sklearn.neighbors import LocalOutlierFactor as SklearnLocalOutlierFactor

from .iforest import IsolationForest
from .lof import LocalOutlierFactor


def compare_lof(reference: np.ndarray, query: np.ndarray, n_neighbors: int, block_size: int) -> dict:
    start = time.perf_counter()
    mine = LocalOutlierFactor(n_neighbors=n_neighbors, block_size=block_size).fit(reference)
    own_scores = mine.score(query)
    own_seconds = time.perf_counter() - start
    start = time.perf_counter()
    theirs = SklearnLocalOutlierFactor(n_neighbors=n_neighbors, novelty=True).fit(reference)
    library_scores = -theirs.score_samples(query)
    library_seconds = time.perf_counter() - start
    return {
        "max_absolute_difference": float(np.abs(own_scores - library_scores).max()),
        "max_relative_difference": float((np.abs(own_scores - library_scores) / np.maximum(np.abs(library_scores), 1e-12)).max()),
        "spearman": float(spearmanr(own_scores, library_scores).statistic),
        "own_seconds": own_seconds,
        "library_seconds": library_seconds,
        "n_reference": int(len(reference)),
        "n_query": int(len(query)),
    }


def compare_iforest(reference: np.ndarray, query: np.ndarray, n_trees: int, subsample_size: int, random_seed: int) -> dict:
    start = time.perf_counter()
    mine = IsolationForest(n_trees=n_trees, subsample_size=subsample_size, random_seed=random_seed).fit(reference)
    own_scores = mine.score(query)
    own_seconds = time.perf_counter() - start
    start = time.perf_counter()
    theirs = SklearnIsolationForest(n_estimators=n_trees, max_samples=subsample_size, random_state=random_seed).fit(reference)
    library_scores = -theirs.score_samples(query)
    library_seconds = time.perf_counter() - start
    top = max(int(0.01 * len(query)), 10)
    own_top = set(np.argsort(-own_scores)[:top].tolist())
    library_top = set(np.argsort(-library_scores)[:top].tolist())
    return {
        "spearman": float(spearmanr(own_scores, library_scores).statistic),
        "top_overlap": len(own_top & library_top) / top,
        "top_k": top,
        "own_seconds": own_seconds,
        "library_seconds": library_seconds,
        "n_reference": int(len(reference)),
        "n_query": int(len(query)),
    }
