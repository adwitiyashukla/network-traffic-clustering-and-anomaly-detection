import numpy as np


class LofError(Exception):
    pass


def _squared_distances(block: np.ndarray, reference: np.ndarray, reference_sq: np.ndarray) -> np.ndarray:
    block_sq = np.einsum("ij,ij->i", block, block)
    products = block @ reference.T
    distances = block_sq[:, None] + reference_sq[None, :] - 2.0 * products
    np.maximum(distances, 0.0, out=distances)
    return distances


class LocalOutlierFactor:
    def __init__(self, n_neighbors: int = 20, block_size: int = 1024):
        if n_neighbors < 1:
            raise LofError("n_neighbors must be at least 1")
        if block_size < 1:
            raise LofError("block_size must be at least 1")
        self.n_neighbors = n_neighbors
        self.block_size = block_size
        self.reference = None
        self.reference_sq = None
        self.k_distance = None
        self.lrd = None

    def _neighbors(self, query: np.ndarray, exclude_self: bool) -> tuple[np.ndarray, np.ndarray]:
        k = self.n_neighbors
        total = len(query)
        indices = np.empty((total, k), dtype=np.int64)
        distances = np.empty((total, k), dtype=np.float64)
        for start in range(0, total, self.block_size):
            stop = min(start + self.block_size, total)
            block = _squared_distances(query[start:stop], self.reference, self.reference_sq)
            if exclude_self:
                rows = np.arange(stop - start)
                block[rows, rows + start] = np.inf
            partition = np.argpartition(block, k - 1, axis=1)[:, :k]
            partition_values = np.take_along_axis(block, partition, axis=1)
            order = np.argsort(partition_values, axis=1)
            chosen = np.take_along_axis(partition, order, axis=1)
            indices[start:stop] = chosen
            distances[start:stop] = np.sqrt(np.take_along_axis(block, chosen, axis=1))
        return indices, distances

    def fit(self, data: np.ndarray) -> "LocalOutlierFactor":
        reference = np.asarray(data, dtype=np.float64)
        if reference.ndim != 2:
            raise LofError("fit expects a two dimensional array")
        if len(reference) <= self.n_neighbors:
            raise LofError("fit needs more than n_neighbors rows")
        self.reference = reference
        self.reference_sq = np.einsum("ij,ij->i", reference, reference)
        indices, distances = self._neighbors(reference, exclude_self=True)
        self.k_distance = distances[:, -1]
        reachability = np.maximum(distances, self.k_distance[indices])
        self.lrd = 1.0 / (reachability.mean(axis=1) + 1e-10)
        return self

    def score(self, data: np.ndarray) -> np.ndarray:
        if self.reference is None:
            raise LofError("score called before fit")
        query = np.asarray(data, dtype=np.float64)
        if query.ndim != 2 or query.shape[1] != self.reference.shape[1]:
            raise LofError("score expects a two dimensional array with the fitted width")
        indices, distances = self._neighbors(query, exclude_self=False)
        reachability = np.maximum(distances, self.k_distance[indices])
        local_density = 1.0 / (reachability.mean(axis=1) + 1e-10)
        return self.lrd[indices].mean(axis=1) / local_density

    def score_fitted(self) -> np.ndarray:
        if self.reference is None:
            raise LofError("score_fitted called before fit")
        indices, distances = self._neighbors(self.reference, exclude_self=True)
        reachability = np.maximum(distances, self.k_distance[indices])
        local_density = 1.0 / (reachability.mean(axis=1) + 1e-10)
        return self.lrd[indices].mean(axis=1) / local_density
