from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from netmine.config import load_config
from netmine.features import FeatureEncoder, FeatureError
from netmine.splits import SplitError, build_evaluation, build_reference, stratified_sample

CATEGORIES = ["Normal", "Generic", "Exploits", "Worms"]


def make_frame(seed=0, rows=2000):
    rng = np.random.default_rng(seed)
    categories = rng.choice(CATEGORIES, size=rows, p=[0.6, 0.2, 0.15, 0.05])
    return pd.DataFrame(
        {
            "id": np.arange(rows),
            "dur": rng.exponential(2.0, size=rows),
            "sbytes": rng.integers(0, 10**6, size=rows),
            "dbytes": rng.integers(0, 10**5, size=rows),
            "rate": rng.exponential(500.0, size=rows),
            "constant": np.zeros(rows),
            "proto": rng.choice(["tcp", "udp", "arp", "ospf"], size=rows, p=[0.5, 0.3, 0.15, 0.05]),
            "service": rng.choice(["-", "dns", "http"], size=rows, p=[0.5, 0.3, 0.2]),
            "state": rng.choice(["fin", "int", "con"], size=rows, p=[0.5, 0.3, 0.2]),
            "attack_cat": categories,
            "label": (categories != "Normal").astype(int),
        }
    )


def small_config():
    config = load_config("config.toml")
    splits = replace(config.splits, reference_size=400, evaluation_normal_size=200)
    features = replace(config.features, rare_category_min_count=50)
    return replace(config, splits=splits, features=features)


def test_encoder_produces_a_finite_matrix():
    config = small_config()
    frame = make_frame()
    matrix = FeatureEncoder(config).fit_transform(frame)
    assert matrix.shape[0] == len(frame)
    assert np.isfinite(matrix).all()
    assert matrix.dtype == np.float32


def test_encoder_drops_constant_columns():
    config = small_config()
    encoder = FeatureEncoder(config).fit(make_frame())
    assert "constant" not in encoder.numeric_columns


def test_encoder_excludes_labels_and_identifiers():
    config = small_config()
    encoder = FeatureEncoder(config).fit(make_frame())
    for column in ["id", "label", "attack_cat", "proto", "service", "state"]:
        assert column not in encoder.numeric_columns


def test_unseen_categories_fall_into_the_rare_bucket():
    config = small_config()
    frame = make_frame()
    encoder = FeatureEncoder(config).fit(frame)
    unseen = frame.head(5).copy()
    unseen["proto"] = "sctp"
    position = encoder.feature_names.index("proto=" + config.features.rare_category_name)
    matrix = encoder.transform(unseen)
    assert (matrix[:, position] == 1.0).all()


def test_extreme_values_are_clipped():
    config = small_config()
    frame = make_frame()
    encoder = FeatureEncoder(config).fit(frame)
    extreme = frame.head(3).copy()
    extreme["rate"] = 10.0**12
    matrix = encoder.transform(extreme)
    assert np.abs(matrix).max() <= config.features.clip_scaled_absolute


def test_transform_before_fit_is_rejected():
    with pytest.raises(FeatureError):
        FeatureEncoder(small_config()).transform(make_frame())


def test_reference_hits_the_requested_contamination():
    config = small_config()
    rng = np.random.default_rng(0)
    for level in [0.0, 0.05, 0.1]:
        reference = build_reference(make_frame(), config, level, rng)
        assert len(reference) == config.splits.reference_size
        assert abs(reference["label"].mean() - level) < 1e-9


def test_evaluation_hits_the_requested_attack_rate():
    config = small_config()
    rng = np.random.default_rng(0)
    for rate in [0.01, 0.05, 0.1]:
        evaluation = build_evaluation(make_frame(), config, rate, rng)
        assert abs(evaluation["label"].mean() - rate) < 0.005


def test_stratified_sample_preserves_proportions():
    rng = np.random.default_rng(1)
    frame = make_frame(rows=4000)
    attacks = frame[frame["attack_cat"] != "Normal"]
    drawn = stratified_sample(attacks, "attack_cat", 400, rng)
    assert len(drawn) == 400
    for category in attacks["attack_cat"].unique():
        expected = (attacks["attack_cat"] == category).mean()
        observed = (drawn["attack_cat"] == category).mean()
        assert abs(observed - expected) < 0.05


def test_stratified_sample_refuses_an_impossible_request():
    rng = np.random.default_rng(1)
    frame = make_frame(rows=100)
    with pytest.raises(SplitError):
        stratified_sample(frame, "attack_cat", 500, rng)


def test_reference_refuses_more_normals_than_exist():
    config = small_config()
    config = replace(config, splits=replace(config.splits, reference_size=100000))
    with pytest.raises(SplitError):
        build_reference(make_frame(), config, 0.0, np.random.default_rng(0))
