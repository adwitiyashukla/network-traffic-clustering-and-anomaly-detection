import numpy as np
import pandas as pd
from scipy.stats import skew
from sklearn.decomposition import PCA

from .config import Config


class FeatureError(Exception):
    pass


class FeatureEncoder:
    def __init__(self, config: Config):
        self.config = config
        self.numeric_columns: list = []
        self.log_columns: list = []
        self.medians: dict = {}
        self.centers: dict = {}
        self.scales: dict = {}
        self.vocabularies: dict = {}
        self.feature_names: list = []
        self.pca = None
        self.fitted = False

    def _excluded(self) -> set:
        features = self.config.features
        return set(features.id_columns) | {features.label_column, features.category_column} | set(features.categorical_columns)

    def _numeric_frame(self, frame: pd.DataFrame) -> pd.DataFrame:
        values = frame[self.numeric_columns].apply(pd.to_numeric, errors="coerce")
        return values.fillna(pd.Series(self.medians))

    def _scaled_numeric(self, frame: pd.DataFrame) -> np.ndarray:
        values = self._numeric_frame(frame)
        for column in self.log_columns:
            values[column] = np.log1p(np.maximum(values[column].to_numpy(dtype=np.float64), 0.0))
        centers = np.array([self.centers[column] for column in self.numeric_columns], dtype=np.float64)
        scales = np.array([self.scales[column] for column in self.numeric_columns], dtype=np.float64)
        return (values.to_numpy(dtype=np.float64) - centers) / scales

    def _one_hot(self, frame: pd.DataFrame) -> np.ndarray:
        blocks = []
        rare = self.config.features.rare_category_name
        for column, vocabulary in self.vocabularies.items():
            raw = frame[column].astype(str).str.strip().str.lower().to_numpy()
            known = set(vocabulary) - {rare}
            mapped = np.where(np.isin(raw, list(known)), raw, rare)
            block = np.zeros((len(frame), len(vocabulary)), dtype=np.float64)
            index = {value: position for position, value in enumerate(vocabulary)}
            block[np.arange(len(frame)), [index[value] for value in mapped]] = 1.0
            blocks.append(block)
        if not blocks:
            return np.zeros((len(frame), 0), dtype=np.float64)
        return np.hstack(blocks)

    def fit(self, frame: pd.DataFrame) -> "FeatureEncoder":
        features = self.config.features
        excluded = self._excluded()
        self.numeric_columns = [column for column in frame.columns if column not in excluded]
        if not self.numeric_columns:
            raise FeatureError("no numeric columns remain after exclusions")
        values = frame[self.numeric_columns].apply(pd.to_numeric, errors="coerce")
        self.medians = values.median().to_dict()
        values = values.fillna(pd.Series(self.medians))
        self.log_columns = []
        for column in self.numeric_columns:
            column_values = values[column].to_numpy(dtype=np.float64)
            if column_values.min() >= 0.0 and abs(float(skew(column_values))) > features.skew_threshold:
                self.log_columns.append(column)
                values[column] = np.log1p(column_values)
        self.centers = {}
        self.scales = {}
        for column in self.numeric_columns:
            column_values = values[column].to_numpy(dtype=np.float64)
            lower, upper = np.percentile(column_values, [25.0, 75.0])
            spread = float(upper - lower)
            if spread <= 0.0:
                spread = float(column_values.std())
            if spread <= 0.0:
                spread = 1.0
            self.centers[column] = float(np.median(column_values))
            self.scales[column] = spread
        scaled = self._scaled_numeric(frame)
        keep = np.var(scaled, axis=0) > features.variance_threshold
        self.numeric_columns = [column for column, flag in zip(self.numeric_columns, keep) if flag]
        if not self.numeric_columns:
            raise FeatureError("every numeric column was dropped by the variance filter")
        self.vocabularies = {}
        for column in features.categorical_columns:
            counts = frame[column].astype(str).str.strip().str.lower().value_counts()
            kept = sorted(counts[counts >= features.rare_category_min_count].index.tolist())
            self.vocabularies[column] = kept + [features.rare_category_name]
        self.feature_names = list(self.numeric_columns)
        for column, vocabulary in self.vocabularies.items():
            self.feature_names.extend([column + "=" + value for value in vocabulary])
        self.fitted = True
        matrix = self._assemble(frame)
        if features.pca_components > 0:
            components = min(features.pca_components, matrix.shape[1], matrix.shape[0])
            self.pca = PCA(n_components=components, whiten=features.pca_whiten, random_state=self.config.splits.random_seed)
            self.pca.fit(matrix)
            self.feature_names = ["pc" + str(index + 1) for index in range(components)]
        return self

    def _assemble(self, frame: pd.DataFrame) -> np.ndarray:
        limit = self.config.features.clip_scaled_absolute
        numeric = np.clip(self._scaled_numeric(frame), -limit, limit)
        categorical = self._one_hot(frame)
        return np.hstack([numeric, categorical]).astype(np.float32)

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        if not self.fitted:
            raise FeatureError("encoder must be fitted before transform")
        matrix = self._assemble(frame)
        if self.pca is not None:
            matrix = self.pca.transform(matrix).astype(np.float32)
        return matrix

    def fit_transform(self, frame: pd.DataFrame) -> np.ndarray:
        return self.fit(frame).transform(frame)
