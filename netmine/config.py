from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib

VALID_COVARIANCE = ("full", "tied", "diag", "spherical")
VALID_ENSEMBLE_MEMBERS = ("lof", "iforest", "cluster_distance", "gmm_nll")


class ConfigError(Exception):
    pass


@dataclass(frozen=True)
class DataConfig:
    train_url: str
    test_url: str
    cache_dir: str
    train_file: str
    test_file: str
    download_timeout: int
    expected_columns: int


@dataclass(frozen=True)
class FeatureConfig:
    id_columns: list
    label_column: str
    category_column: str
    normal_category: str
    categorical_columns: list
    rare_category_min_count: int
    rare_category_name: str
    skew_threshold: float
    variance_threshold: float
    clip_scaled_absolute: float
    pca_components: int
    pca_whiten: bool


@dataclass(frozen=True)
class SplitConfig:
    random_seed: int
    reference_size: int
    evaluation_normal_size: int
    evaluation_attack_rates: list
    reference_contamination_rates: list
    primary_attack_rate: float
    include_natural_rate: bool


@dataclass(frozen=True)
class ClusterConfig:
    kmeans_k_grid: list
    kmeans_n_init: int
    kmeans_max_iter: int
    silhouette_sample_size: int
    dbscan_eps: float
    dbscan_min_samples: int
    dbscan_sample_size: int
    gmm_components: int
    gmm_covariance_type: str
    gmm_reg_covar: float


@dataclass(frozen=True)
class LofConfig:
    n_neighbors: int
    block_size: int
    fit_sample_size: int
    deduplicate: bool


@dataclass(frozen=True)
class IforestConfig:
    n_trees: int
    subsample_size: int
    random_seed: int


@dataclass(frozen=True)
class EnsembleConfig:
    members: list


@dataclass(frozen=True)
class EvaluateConfig:
    precision_at_k: list
    fpr_grid: list
    verification_sample_size: int
    output_dir: str


@dataclass(frozen=True)
class Config:
    root: Path
    data: DataConfig
    features: FeatureConfig
    splits: SplitConfig
    cluster: ClusterConfig
    lof: LofConfig
    iforest: IforestConfig
    ensemble: EnsembleConfig
    evaluate: EvaluateConfig

    @property
    def cache_dir(self) -> Path:
        return self.root / self.data.cache_dir

    @property
    def output_dir(self) -> Path:
        return self.root / self.evaluate.output_dir


def _section(raw: dict, name: str) -> dict:
    if name not in raw:
        raise ConfigError("missing section [" + name + "]")
    value = raw[name]
    if not isinstance(value, dict):
        raise ConfigError("section [" + name + "] must be a table")
    return value


def _build(cls, raw: dict, name: str):
    expected = {f.name for f in fields(cls)}
    present = set(raw)
    unknown = sorted(present - expected)
    if unknown:
        raise ConfigError("unknown keys in [" + name + "]: " + ", ".join(unknown))
    missing = sorted(expected - present)
    if missing:
        raise ConfigError("missing keys in [" + name + "]: " + ", ".join(missing))
    return cls(**raw)


def _check(condition: bool, message: str) -> None:
    if not condition:
        raise ConfigError(message)


def _all_str(values: Any) -> bool:
    return isinstance(values, list) and all(isinstance(v, str) for v in values)


def _all_positive_int(values: Any) -> bool:
    return isinstance(values, list) and len(values) > 0 and all(
        isinstance(v, int) and not isinstance(v, bool) and v > 0 for v in values
    )


def validate(config: Config) -> None:
    data = config.data
    _check(data.train_url.startswith("http"), "data.train_url must be an http url")
    _check(data.test_url.startswith("http"), "data.test_url must be an http url")
    _check(data.download_timeout > 0, "data.download_timeout must be positive")
    _check(data.expected_columns > 1, "data.expected_columns must be greater than 1")

    features = config.features
    _check(_all_str(features.id_columns), "features.id_columns must be a list of strings")
    _check(_all_str(features.categorical_columns), "features.categorical_columns must be a list of strings")
    _check(len(features.label_column) > 0, "features.label_column must not be empty")
    _check(len(features.category_column) > 0, "features.category_column must not be empty")
    _check(features.rare_category_min_count >= 0, "features.rare_category_min_count must not be negative")
    _check(features.skew_threshold >= 0, "features.skew_threshold must not be negative")
    _check(features.variance_threshold > 0, "features.variance_threshold must be positive")
    _check(features.clip_scaled_absolute > 0, "features.clip_scaled_absolute must be positive")
    _check(features.pca_components >= 0, "features.pca_components must not be negative")

    splits = config.splits
    _check(splits.reference_size > 0, "splits.reference_size must be positive")
    _check(splits.evaluation_normal_size > 0, "splits.evaluation_normal_size must be positive")
    _check(len(splits.evaluation_attack_rates) > 0, "splits.evaluation_attack_rates must not be empty")
    for rate in splits.evaluation_attack_rates:
        _check(isinstance(rate, float) and 0.0 < rate < 1.0, "splits.evaluation_attack_rates must be floats in (0, 1)")
    _check(len(splits.reference_contamination_rates) > 0, "splits.reference_contamination_rates must not be empty")
    for rate in splits.reference_contamination_rates:
        _check(isinstance(rate, float) and 0.0 <= rate < 1.0, "splits.reference_contamination_rates must be floats in [0, 1)")
    _check(0.0 < splits.primary_attack_rate < 1.0, "splits.primary_attack_rate must be in (0, 1)")
    _check(splits.primary_attack_rate in splits.evaluation_attack_rates, "splits.primary_attack_rate must appear in splits.evaluation_attack_rates")

    cluster = config.cluster
    _check(_all_positive_int(cluster.kmeans_k_grid), "cluster.kmeans_k_grid must be a non empty list of positive ints")
    _check(min(cluster.kmeans_k_grid) >= 2, "cluster.kmeans_k_grid values must be at least 2")
    _check(cluster.kmeans_n_init > 0, "cluster.kmeans_n_init must be positive")
    _check(cluster.kmeans_max_iter > 0, "cluster.kmeans_max_iter must be positive")
    _check(cluster.silhouette_sample_size >= 100, "cluster.silhouette_sample_size must be at least 100")
    _check(cluster.dbscan_eps > 0, "cluster.dbscan_eps must be positive")
    _check(cluster.dbscan_min_samples > 0, "cluster.dbscan_min_samples must be positive")
    _check(cluster.dbscan_sample_size > cluster.dbscan_min_samples, "cluster.dbscan_sample_size must exceed dbscan_min_samples")
    _check(cluster.gmm_components >= 1, "cluster.gmm_components must be at least 1")
    _check(cluster.gmm_covariance_type in VALID_COVARIANCE, "cluster.gmm_covariance_type must be one of " + ", ".join(VALID_COVARIANCE))
    _check(cluster.gmm_reg_covar > 0, "cluster.gmm_reg_covar must be positive")

    lof = config.lof
    _check(lof.n_neighbors >= 1, "lof.n_neighbors must be at least 1")
    _check(lof.block_size >= 1, "lof.block_size must be at least 1")
    _check(lof.fit_sample_size > lof.n_neighbors, "lof.fit_sample_size must exceed lof.n_neighbors")

    iforest = config.iforest
    _check(iforest.n_trees >= 1, "iforest.n_trees must be at least 1")
    _check(iforest.subsample_size >= 4, "iforest.subsample_size must be at least 4")

    ensemble = config.ensemble
    _check(len(ensemble.members) > 0, "ensemble.members must not be empty")
    unknown = sorted(set(ensemble.members) - set(VALID_ENSEMBLE_MEMBERS))
    _check(not unknown, "unknown ensemble.members: " + ", ".join(unknown))

    evaluate = config.evaluate
    _check(_all_positive_int(evaluate.precision_at_k), "evaluate.precision_at_k must be a non empty list of positive ints")
    _check(len(evaluate.fpr_grid) > 0, "evaluate.fpr_grid must not be empty")
    for rate in evaluate.fpr_grid:
        _check(isinstance(rate, float) and 0.0 < rate < 1.0, "evaluate.fpr_grid must be floats in (0, 1)")
    _check(evaluate.verification_sample_size >= 100, "evaluate.verification_sample_size must be at least 100")
    _check(len(evaluate.output_dir) > 0, "evaluate.output_dir must not be empty")


def load_config(path: str | Path) -> Config:
    path = Path(path)
    if not path.exists():
        raise ConfigError("config file not found: " + str(path))
    with open(path, "rb") as handle:
        raw = tomllib.load(handle)
    known = {"data", "features", "splits", "cluster", "lof", "iforest", "ensemble", "evaluate"}
    unknown = sorted(set(raw) - known)
    if unknown:
        raise ConfigError("unknown sections: " + ", ".join(unknown))
    config = Config(
        root=path.resolve().parent,
        data=_build(DataConfig, _section(raw, "data"), "data"),
        features=_build(FeatureConfig, _section(raw, "features"), "features"),
        splits=_build(SplitConfig, _section(raw, "splits"), "splits"),
        cluster=_build(ClusterConfig, _section(raw, "cluster"), "cluster"),
        lof=_build(LofConfig, _section(raw, "lof"), "lof"),
        iforest=_build(IforestConfig, _section(raw, "iforest"), "iforest"),
        ensemble=_build(EnsembleConfig, _section(raw, "ensemble"), "ensemble"),
        evaluate=_build(EvaluateConfig, _section(raw, "evaluate"), "evaluate"),
    )
    validate(config)
    return config
