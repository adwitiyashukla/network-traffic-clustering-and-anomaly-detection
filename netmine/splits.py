import numpy as np
import pandas as pd

from .config import Config


class SplitError(Exception):
    pass


def _largest_remainder(proportions: np.ndarray, total: int) -> np.ndarray:
    raw = proportions * total
    counts = np.floor(raw).astype(int)
    shortfall = total - int(counts.sum())
    if shortfall > 0:
        order = np.argsort(-(raw - counts))
        counts[order[:shortfall]] += 1
    return counts


def stratified_sample(frame: pd.DataFrame, column: str, total: int, rng: np.random.Generator) -> pd.DataFrame:
    if total <= 0:
        return frame.iloc[0:0]
    if total > len(frame):
        raise SplitError("requested " + str(total) + " rows from a pool of " + str(len(frame)))
    groups = frame.groupby(column, sort=True)
    keys = list(groups.groups.keys())
    sizes = np.array([len(groups.groups[key]) for key in keys], dtype=float)
    counts = np.minimum(_largest_remainder(sizes / sizes.sum(), total), sizes.astype(int))
    deficit = total - int(counts.sum())
    while deficit > 0:
        room = np.array([sizes[i] - counts[i] for i in range(len(keys))])
        if room.max() <= 0:
            raise SplitError("cannot draw " + str(total) + " rows while preserving strata")
        take = int(np.argmax(room))
        counts[take] += 1
        deficit -= 1
    pieces = []
    for index, key in enumerate(keys):
        if counts[index] == 0:
            continue
        member = frame.loc[groups.groups[key]]
        picked = rng.choice(len(member), size=int(counts[index]), replace=False)
        pieces.append(member.iloc[np.sort(picked)])
    return pd.concat(pieces).sample(frac=1.0, random_state=int(rng.integers(0, 2**31 - 1)))


def build_reference(train: pd.DataFrame, config: Config, contamination: float, rng: np.random.Generator) -> pd.DataFrame:
    features = config.features
    size = config.splits.reference_size
    normal_pool = train[train[features.category_column] == features.normal_category]
    attack_pool = train[train[features.category_column] != features.normal_category]
    n_attack = int(round(size * contamination))
    n_normal = size - n_attack
    if n_normal > len(normal_pool):
        raise SplitError("reference_size needs " + str(n_normal) + " normal flows but only " + str(len(normal_pool)) + " exist")
    if n_attack > len(attack_pool):
        raise SplitError("reference contamination needs " + str(n_attack) + " attack flows but only " + str(len(attack_pool)) + " exist")
    normal_part = normal_pool.iloc[np.sort(rng.choice(len(normal_pool), size=n_normal, replace=False))]
    if n_attack == 0:
        combined = normal_part
    else:
        attack_part = stratified_sample(attack_pool, features.category_column, n_attack, rng)
        combined = pd.concat([normal_part, attack_part])
    return combined.sample(frac=1.0, random_state=int(rng.integers(0, 2**31 - 1))).reset_index(drop=True)


def build_evaluation(test: pd.DataFrame, config: Config, attack_rate: float, rng: np.random.Generator) -> pd.DataFrame:
    features = config.features
    normal_pool = test[test[features.category_column] == features.normal_category]
    attack_pool = test[test[features.category_column] != features.normal_category]
    n_normal = min(config.splits.evaluation_normal_size, len(normal_pool))
    n_attack = int(round(n_normal * attack_rate / (1.0 - attack_rate)))
    if n_attack > len(attack_pool):
        raise SplitError("attack rate " + str(attack_rate) + " needs " + str(n_attack) + " attack flows but only " + str(len(attack_pool)) + " exist")
    normal_part = normal_pool.iloc[np.sort(rng.choice(len(normal_pool), size=n_normal, replace=False))]
    attack_part = stratified_sample(attack_pool, features.category_column, n_attack, rng)
    combined = pd.concat([normal_part, attack_part])
    return combined.sample(frac=1.0, random_state=int(rng.integers(0, 2**31 - 1))).reset_index(drop=True)


