import numpy as np

EULER_MASCHERONI = 0.5772156649015329


class IsolationForestError(Exception):
    pass


def average_path_length(sizes: np.ndarray) -> np.ndarray:
    counts = np.asarray(sizes, dtype=np.float64)
    result = np.zeros_like(counts)
    large = counts > 2.0
    result[large] = 2.0 * (np.log(counts[large] - 1.0) + EULER_MASCHERONI) - 2.0 * (counts[large] - 1.0) / counts[large]
    result[counts == 2.0] = 1.0
    return result


class _Tree:
    def __init__(self):
        self.feature: list = []
        self.threshold: list = []
        self.left: list = []
        self.right: list = []
        self.size: list = []
        self.depth: list = []

    def add(self) -> int:
        self.feature.append(-1)
        self.threshold.append(0.0)
        self.left.append(-1)
        self.right.append(-1)
        self.size.append(0)
        self.depth.append(0)
        return len(self.feature) - 1

    def freeze(self) -> None:
        self.feature = np.array(self.feature, dtype=np.int64)
        self.threshold = np.array(self.threshold, dtype=np.float64)
        self.left = np.array(self.left, dtype=np.int64)
        self.right = np.array(self.right, dtype=np.int64)
        self.size = np.array(self.size, dtype=np.float64)
        self.depth = np.array(self.depth, dtype=np.float64)


def _grow(sample: np.ndarray, height_limit: int, rng: np.random.Generator) -> _Tree:
    tree = _Tree()
    root = tree.add()
    stack = [(root, np.arange(len(sample)), 0)]
    while stack:
        node, rows, depth = stack.pop()
        tree.depth[node] = depth
        tree.size[node] = len(rows)
        if depth >= height_limit or len(rows) <= 1:
            continue
        values = sample[rows]
        lower = values.min(axis=0)
        upper = values.max(axis=0)
        usable = np.flatnonzero(upper > lower)
        if usable.size == 0:
            continue
        feature = int(rng.choice(usable))
        threshold = float(rng.uniform(lower[feature], upper[feature]))
        mask = values[:, feature] < threshold
        if mask.all() or not mask.any():
            continue
        left = tree.add()
        right = tree.add()
        tree.feature[node] = feature
        tree.threshold[node] = threshold
        tree.left[node] = left
        tree.right[node] = right
        stack.append((left, rows[mask], depth + 1))
        stack.append((right, rows[~mask], depth + 1))
    tree.freeze()
    return tree


class IsolationForest:
    def __init__(self, n_trees: int = 150, subsample_size: int = 256, random_seed: int = 0):
        if n_trees < 1:
            raise IsolationForestError("n_trees must be at least 1")
        if subsample_size < 4:
            raise IsolationForestError("subsample_size must be at least 4")
        self.n_trees = n_trees
        self.subsample_size = subsample_size
        self.random_seed = random_seed
        self.trees: list = []
        self.height_limit = 0
        self.normaliser = 1.0

    def fit(self, data: np.ndarray) -> "IsolationForest":
        matrix = np.asarray(data, dtype=np.float64)
        if matrix.ndim != 2:
            raise IsolationForestError("fit expects a two dimensional array")
        if len(matrix) < 4:
            raise IsolationForestError("fit needs at least four rows")
        rng = np.random.default_rng(self.random_seed)
        size = min(self.subsample_size, len(matrix))
        self.height_limit = int(np.ceil(np.log2(max(size, 2))))
        self.normaliser = float(average_path_length(np.array([size]))[0])
        if self.normaliser <= 0.0:
            self.normaliser = 1.0
        self.trees = []
        for _ in range(self.n_trees):
            rows = rng.choice(len(matrix), size=size, replace=False)
            self.trees.append(_grow(matrix[rows], self.height_limit, rng))
        return self

    def _tree_path(self, tree: _Tree, matrix: np.ndarray) -> np.ndarray:
        current = np.zeros(len(matrix), dtype=np.int64)
        for _ in range(self.height_limit + 1):
            internal = np.flatnonzero(tree.feature[current] >= 0)
            if internal.size == 0:
                break
            nodes = current[internal]
            features = tree.feature[nodes]
            thresholds = tree.threshold[nodes]
            go_left = matrix[internal, features] < thresholds
            current[internal] = np.where(go_left, tree.left[nodes], tree.right[nodes])
        return tree.depth[current] + average_path_length(tree.size[current])

    def score(self, data: np.ndarray) -> np.ndarray:
        if not self.trees:
            raise IsolationForestError("score called before fit")
        matrix = np.asarray(data, dtype=np.float64)
        if matrix.ndim != 2:
            raise IsolationForestError("score expects a two dimensional array")
        total = np.zeros(len(matrix), dtype=np.float64)
        for tree in self.trees:
            total += self._tree_path(tree, matrix)
        return np.power(2.0, -(total / len(self.trees)) / self.normaliser)
