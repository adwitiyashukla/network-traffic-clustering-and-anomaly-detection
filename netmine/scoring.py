import time

import numpy as np

from .cluster import fit_gmm, fit_kmeans, select_k
from .config import Config
from .iforest import IsolationForest
from .lof import LocalOutlierFactor


class ScoringError(Exception):
    pass


def rank_normalise(scores: np.ndarray) -> np.ndarray:
    values = np.asarray(scores, dtype=np.float64)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    ranks[order] = np.arange(len(values), dtype=np.float64)
    unique, inverse, counts = np.unique(values, return_inverse=True, return_counts=True)
    sums = np.zeros(len(unique), dtype=np.float64)
    np.add.at(sums, inverse, ranks)
    ranks = (sums / counts)[inverse]
    if len(values) < 2:
        return np.zeros(len(values), dtype=np.float64)
    return ranks / (len(values) - 1.0)


def lof_reference(matrix: np.ndarray, config: Config, rng: np.random.Generator) -> np.ndarray:
    pool = np.unique(matrix, axis=0) if config.lof.deduplicate else matrix
    size = min(config.lof.fit_sample_size, len(pool))
    rows = np.sort(rng.choice(len(pool), size=size, replace=False))
    return pool[rows]


class ClusterDistance:
    def __init__(self, config: Config):
        self.config = config
        self.kmeans = None
        self.k = 0
        self.reference_scale = None

    def fit(self, matrix: np.ndarray) -> "ClusterDistance":
        self.k, _ = select_k(matrix, self.config)
        self.kmeans = fit_kmeans(matrix, self.k, self.config)
        distances = np.linalg.norm(matrix - self.kmeans.cluster_centers_[self.kmeans.labels_], axis=1)
        scale = np.ones(self.k, dtype=np.float64)
        for cluster in range(self.k):
            members = distances[self.kmeans.labels_ == cluster]
            if members.size > 0:
                median = float(np.median(members))
                scale[cluster] = median if median > 0.0 else 1.0
        self.reference_scale = scale
        return self

    def score(self, matrix: np.ndarray) -> np.ndarray:
        if self.kmeans is None:
            raise ScoringError("score called before fit")
        assignments = self.kmeans.predict(matrix)
        distances = np.linalg.norm(matrix - self.kmeans.cluster_centers_[assignments], axis=1)
        return distances / self.reference_scale[assignments]


class GmmDensity:
    def __init__(self, config: Config):
        self.config = config
        self.model = None

    def fit(self, matrix: np.ndarray) -> "GmmDensity":
        self.model = fit_gmm(matrix, self.config)
        return self

    def score(self, matrix: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise ScoringError("score called before fit")
        return -self.model.score_samples(matrix)


class DetectorBank:
    def __init__(self, config: Config):
        self.config = config
        self.lof = None
        self.iforest = None
        self.cluster_distance = None
        self.gmm = None
        self.fit_seconds: dict = {}
        self.score_seconds: dict = {}
        self.lof_rows = None

    def fit(self, matrix: np.ndarray, rng: np.random.Generator) -> "DetectorBank":
        settings = self.config
        self.lof_rows = lof_reference(matrix, settings, rng)
        start = time.perf_counter()
        self.lof = LocalOutlierFactor(settings.lof.n_neighbors, settings.lof.block_size).fit(self.lof_rows)
        self.fit_seconds["lof"] = time.perf_counter() - start
        start = time.perf_counter()
        self.iforest = IsolationForest(settings.iforest.n_trees, settings.iforest.subsample_size, settings.iforest.random_seed).fit(matrix)
        self.fit_seconds["iforest"] = time.perf_counter() - start
        start = time.perf_counter()
        self.cluster_distance = ClusterDistance(settings).fit(matrix)
        self.fit_seconds["cluster_distance"] = time.perf_counter() - start
        start = time.perf_counter()
        self.gmm = GmmDensity(settings).fit(matrix)
        self.fit_seconds["gmm_nll"] = time.perf_counter() - start
        return self

    def score(self, matrix: np.ndarray) -> dict:
        scores = {}
        timings = {}
        for name, detector in (
            ("lof", self.lof),
            ("iforest", self.iforest),
            ("cluster_distance", self.cluster_distance),
            ("gmm_nll", self.gmm),
        ):
            if detector is None:
                raise ScoringError("score called before fit")
            start = time.perf_counter()
            scores[name] = detector.score(matrix)
            timings[name] = time.perf_counter() - start
        members = self.config.ensemble.members
        stacked = np.vstack([rank_normalise(scores[name]) for name in members])
        scores["ensemble"] = stacked.mean(axis=0)
        timings["ensemble"] = 0.0
        self.score_seconds = timings
        return scores
