import numpy as np
from sklearn.cluster import DBSCAN, KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.mixture import GaussianMixture

from .config import Config


class ClusterError(Exception):
    pass


def purity(labels: np.ndarray, truth: np.ndarray) -> float:
    total = 0
    for cluster in np.unique(labels):
        members = truth[labels == cluster]
        if members.size == 0:
            continue
        values, counts = np.unique(members, return_counts=True)
        total += int(counts.max())
    return total / len(truth)


def _silhouette(matrix: np.ndarray, labels: np.ndarray, sample_size: int, seed: int) -> float:
    unique = np.unique(labels)
    if unique.size < 2:
        return float("nan")
    size = min(sample_size, len(matrix))
    return float(silhouette_score(matrix, labels, sample_size=size, random_state=seed))


def select_k(matrix: np.ndarray, config: Config) -> tuple:
    settings = config.cluster
    seed = config.splits.random_seed
    table = []
    for k in settings.kmeans_k_grid:
        model = KMeans(n_clusters=k, n_init=settings.kmeans_n_init, max_iter=settings.kmeans_max_iter, random_state=seed)
        labels = model.fit_predict(matrix)
        table.append(
            {
                "k": k,
                "inertia": float(model.inertia_),
                "silhouette": _silhouette(matrix, labels, settings.silhouette_sample_size, seed),
            }
        )
    best = max(table, key=lambda row: -1.0 if np.isnan(row["silhouette"]) else row["silhouette"])
    return int(best["k"]), table


def fit_kmeans(matrix: np.ndarray, k: int, config: Config) -> KMeans:
    settings = config.cluster
    return KMeans(
        n_clusters=k,
        n_init=settings.kmeans_n_init,
        max_iter=settings.kmeans_max_iter,
        random_state=config.splits.random_seed,
    ).fit(matrix)


def fit_gmm(matrix: np.ndarray, config: Config) -> GaussianMixture:
    settings = config.cluster
    return GaussianMixture(
        n_components=settings.gmm_components,
        covariance_type=settings.gmm_covariance_type,
        reg_covar=settings.gmm_reg_covar,
        random_state=config.splits.random_seed,
    ).fit(matrix)


def structure_report(matrix: np.ndarray, categories: np.ndarray, labels: np.ndarray, config: Config) -> dict:
    settings = config.cluster
    seed = config.splits.random_seed
    rng = np.random.default_rng(seed)
    best_k, table = select_k(matrix, config)
    kmeans = fit_kmeans(matrix, best_k, config)
    assignments = kmeans.labels_
    report = {
        "kmeans_grid": table,
        "kmeans_k": best_k,
        "kmeans_silhouette": _silhouette(matrix, assignments, settings.silhouette_sample_size, seed),
        "kmeans_purity_category": purity(assignments, categories),
        "kmeans_ari_category": float(adjusted_rand_score(categories, assignments)),
        "kmeans_ari_binary": float(adjusted_rand_score(labels, assignments)),
        "kmeans_cluster_attack_rate": [],
    }
    for cluster in range(best_k):
        mask = assignments == cluster
        report["kmeans_cluster_attack_rate"].append(
            {"cluster": cluster, "size": int(mask.sum()), "attack_rate": float(labels[mask].mean()) if mask.any() else 0.0}
        )
    sample_size = min(settings.dbscan_sample_size, len(matrix))
    rows = np.sort(rng.choice(len(matrix), size=sample_size, replace=False))
    dbscan_labels = DBSCAN(eps=settings.dbscan_eps, min_samples=settings.dbscan_min_samples).fit_predict(matrix[rows])
    noise = dbscan_labels == -1
    report["dbscan"] = {
        "sample_size": int(sample_size),
        "clusters": int(len(set(dbscan_labels.tolist())) - (1 if noise.any() else 0)),
        "noise_fraction": float(noise.mean()),
        "noise_attack_rate": float(labels[rows][noise].mean()) if noise.any() else 0.0,
        "core_attack_rate": float(labels[rows][~noise].mean()) if (~noise).any() else 0.0,
        "base_attack_rate": float(labels[rows].mean()),
        "ari_binary": float(adjusted_rand_score(labels[rows], dbscan_labels)),
    }
    return report
