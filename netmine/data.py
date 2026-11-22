import urllib.error
import urllib.request
from pathlib import Path

import pandas as pd

from .config import Config

CHUNK_SIZE = 1 << 20


class DataError(Exception):
    pass


def _download(url: str, destination: Path, timeout: int) -> None:
    partial = destination.with_suffix(destination.suffix + ".part")
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            if response.status != 200:
                raise DataError("download failed with status " + str(response.status) + " for " + url)
            with open(partial, "wb") as handle:
                while True:
                    chunk = response.read(CHUNK_SIZE)
                    if not chunk:
                        break
                    handle.write(chunk)
    except urllib.error.URLError as error:
        if partial.exists():
            partial.unlink()
        raise DataError("could not download " + url + ": " + str(error.reason))
    if partial.stat().st_size == 0:
        partial.unlink()
        raise DataError("downloaded an empty file from " + url)
    partial.replace(destination)


def ensure_dataset(config: Config) -> tuple[Path, Path]:
    cache = config.cache_dir
    cache.mkdir(parents=True, exist_ok=True)
    train_path = cache / config.data.train_file
    test_path = cache / config.data.test_file
    for path, url in ((train_path, config.data.train_url), (test_path, config.data.test_url)):
        if not path.exists():
            _download(url, path, config.data.download_timeout)
    return train_path, test_path


def _validate_frame(frame: pd.DataFrame, name: str, config: Config) -> None:
    features = config.features
    if frame.shape[1] != config.data.expected_columns:
        raise DataError(name + " has " + str(frame.shape[1]) + " columns, expected " + str(config.data.expected_columns))
    if frame.shape[0] == 0:
        raise DataError(name + " is empty")
    required = [features.label_column, features.category_column] + list(features.categorical_columns)
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise DataError(name + " is missing columns: " + ", ".join(missing))
    labels = set(frame[features.label_column].unique().tolist())
    if not labels.issubset({0, 1}):
        raise DataError(name + " has labels outside {0, 1}: " + str(sorted(labels)))
    if frame[features.category_column].isna().any():
        raise DataError(name + " has missing values in " + features.category_column)
    if features.normal_category not in set(frame[features.category_column].unique().tolist()):
        raise DataError(name + " has no rows in category " + features.normal_category)
    normal_mask = frame[features.category_column] == features.normal_category
    label_mask = frame[features.label_column] == 0
    if not normal_mask.equals(label_mask):
        raise DataError(name + " disagrees between " + features.category_column + " and " + features.label_column)


def load_frames(config: Config) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_path, test_path = ensure_dataset(config)
    train = pd.read_csv(train_path, encoding="utf-8-sig", low_memory=False)
    test = pd.read_csv(test_path, encoding="utf-8-sig", low_memory=False)
    for frame in (train, test):
        frame.columns = [column.strip() for column in frame.columns]
        for column in config.features.categorical_columns:
            if column in frame.columns:
                frame[column] = frame[column].astype(str).str.strip().str.lower()
        frame[config.features.category_column] = frame[config.features.category_column].astype(str).str.strip()
    _validate_frame(train, "training set", config)
    _validate_frame(test, "testing set", config)
    return train, test
