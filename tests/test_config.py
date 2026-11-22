import pytest

from netmine.config import ConfigError, load_config

BASE = """
[data]
train_url = "https://example.com/train.csv"
test_url = "https://example.com/test.csv"
cache_dir = "data"
train_file = "train.csv"
test_file = "test.csv"
download_timeout = 60
expected_columns = 45

[features]
id_columns = ["id"]
label_column = "label"
category_column = "attack_cat"
normal_category = "Normal"
categorical_columns = ["proto"]
rare_category_min_count = 10
rare_category_name = "other"
skew_threshold = 2.0
variance_threshold = 1e-8
clip_scaled_absolute = 20.0
pca_components = 0
pca_whiten = false

[splits]
random_seed = 1
reference_size = 100
evaluation_normal_size = 50
evaluation_attack_rates = [0.05]
reference_contamination_rates = [0.0]
primary_attack_rate = 0.05
include_natural_rate = true

[cluster]
kmeans_k_grid = [2, 3]
kmeans_n_init = 2
kmeans_max_iter = 50
silhouette_sample_size = 100
dbscan_eps = 1.0
dbscan_min_samples = 5
dbscan_sample_size = 50
gmm_components = 2
gmm_covariance_type = "full"
gmm_reg_covar = 0.0001

[lof]
n_neighbors = 5
block_size = 32
fit_sample_size = 50
deduplicate = true

[iforest]
n_trees = 10
subsample_size = 32
random_seed = 1

[ensemble]
members = ["lof", "iforest"]

[evaluate]
precision_at_k = [10]
fpr_grid = [0.05]
verification_sample_size = 100
output_dir = "results"
"""


def write(tmp_path, text):
    path = tmp_path / "config.toml"
    path.write_text(text, encoding="utf-8")
    return path


def test_repository_config_loads():
    config = load_config("config.toml")
    assert config.data.expected_columns == 45
    assert config.splits.primary_attack_rate in config.splits.evaluation_attack_rates


def test_valid_config_loads(tmp_path):
    config = load_config(write(tmp_path, BASE))
    assert config.lof.n_neighbors == 5
    assert config.ensemble.members == ["lof", "iforest"]


def test_unknown_key_is_rejected(tmp_path):
    with pytest.raises(ConfigError, match="unknown keys"):
        load_config(write(tmp_path, BASE.replace("expected_columns = 45", "expected_columns = 45\nextra = 1", 1)))


def test_missing_key_is_rejected(tmp_path):
    text = BASE.replace("n_trees = 10\n", "")
    with pytest.raises(ConfigError, match="missing keys"):
        load_config(write(tmp_path, text))


def test_out_of_range_value_is_rejected(tmp_path):
    text = BASE.replace("subsample_size = 32", "subsample_size = 2")
    with pytest.raises(ConfigError, match="subsample_size"):
        load_config(write(tmp_path, text))


def test_primary_rate_must_be_evaluated(tmp_path):
    text = BASE.replace("primary_attack_rate = 0.05", "primary_attack_rate = 0.2")
    with pytest.raises(ConfigError, match="primary_attack_rate"):
        load_config(write(tmp_path, text))


def test_unknown_ensemble_member_is_rejected(tmp_path):
    text = BASE.replace('members = ["lof", "iforest"]', 'members = ["lof", "magic"]')
    with pytest.raises(ConfigError, match="magic"):
        load_config(write(tmp_path, text))


def test_missing_file_is_rejected(tmp_path):
    with pytest.raises(ConfigError, match="not found"):
        load_config(tmp_path / "absent.toml")
