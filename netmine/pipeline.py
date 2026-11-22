import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .cluster import structure_report
from .config import Config
from .data import load_frames
from .evaluate import detection_report, ranking_metrics
from .features import FeatureEncoder
from .scoring import DetectorBank
from .splits import build_evaluation, build_reference
from .verify import compare_iforest, compare_lof

DETECTORS = ("lof", "iforest", "cluster_distance", "gmm_nll", "ensemble")


def _log(message: str, verbose: bool) -> None:
    if verbose:
        print(message, flush=True)


def _fit_stack(frame: pd.DataFrame, config: Config, rng: np.random.Generator) -> tuple:
    encoder = FeatureEncoder(config).fit(frame)
    matrix = encoder.transform(frame)
    bank = DetectorBank(config).fit(matrix, rng)
    return encoder, bank


def run(config: Config, verbose: bool = True) -> dict:
    started = time.perf_counter()
    features = config.features
    splits = config.splits
    rng = np.random.default_rng(splits.random_seed)

    _log("loading data", verbose)
    train, test = load_frames(config)

    _log("fitting detectors on a clean reference of " + str(splits.reference_size) + " flows", verbose)
    reference = build_reference(train, config, 0.0, rng)
    encoder, bank = _fit_stack(reference, config, rng)

    evaluation_sets = {}
    for rate in splits.evaluation_attack_rates:
        evaluation_sets[format(rate, ".2f")] = build_evaluation(test, config, rate, rng)
    if splits.include_natural_rate:
        evaluation_sets["natural"] = test.reset_index(drop=True)

    base_rate_rows = []
    natural_scores = None
    natural_score_seconds = {}
    primary_key = format(splits.primary_attack_rate, ".2f")
    for name, frame in evaluation_sets.items():
        _log("scoring evaluation set " + name + " with " + str(len(frame)) + " flows", verbose)
        matrix = encoder.transform(frame)
        labels = frame[features.label_column].to_numpy()
        scores = bank.score(matrix)
        if name == "natural":
            natural_scores = (scores, labels, frame[features.category_column].to_numpy())
            natural_score_seconds = dict(bank.score_seconds)
        for detector in DETECTORS:
            row = {"evaluation_set": name, "detector": detector}
            row.update(ranking_metrics(scores[detector], labels, config.evaluate.precision_at_k))
            base_rate_rows.append(row)

    _log("measuring per category recall at fixed false positive rates", verbose)
    category_rows = []
    scores, labels, categories = natural_scores
    for detector in DETECTORS:
        for entry in detection_report(scores[detector], labels, categories, config.evaluate.fpr_grid, features.normal_category):
            for category, values in entry["per_category"].items():
                category_rows.append(
                    {
                        "detector": detector,
                        "target_fpr": entry["target_fpr"],
                        "observed_fpr": entry["observed_fpr"],
                        "overall_recall": entry["overall_recall"],
                        "category": category,
                        "support": values["support"],
                        "recall": values["recall"],
                    }
                )

    _log("running the baseline contamination experiment", verbose)
    primary = evaluation_sets[primary_key]
    primary_labels = primary[features.label_column].to_numpy()
    contamination_rows = []
    for level in splits.reference_contamination_rates:
        if level == 0.0:
            level_encoder, level_bank = encoder, bank
        else:
            level_reference = build_reference(train, config, level, rng)
            level_encoder, level_bank = _fit_stack(level_reference, config, rng)
        level_scores = level_bank.score(level_encoder.transform(primary))
        for detector in DETECTORS:
            row = {"contamination": level, "detector": detector}
            row.update(ranking_metrics(level_scores[detector], primary_labels, config.evaluate.precision_at_k))
            contamination_rows.append(row)
        _log("  contamination " + format(level, ".2f") + " done", verbose)

    _log("describing cluster structure of the evaluation traffic", verbose)
    primary_matrix = encoder.transform(primary)
    structure = structure_report(
        primary_matrix,
        primary[features.category_column].to_numpy(),
        primary_labels,
        config,
    )

    _log("cross checking the from scratch detectors against scikit-learn", verbose)
    sample = min(config.evaluate.verification_sample_size, len(primary_matrix))
    reference_matrix = encoder.transform(reference)
    verification = {
        "lof": compare_lof(
            bank.lof_rows,
            primary_matrix[:sample],
            config.lof.n_neighbors,
            config.lof.block_size,
        ),
        "iforest": compare_iforest(
            reference_matrix,
            primary_matrix[:sample],
            config.iforest.n_trees,
            config.iforest.subsample_size,
            config.iforest.random_seed,
        ),
    }

    summary = {
        "seconds": time.perf_counter() - started,
        "n_features": int(reference_matrix.shape[1]),
        "feature_names": encoder.feature_names,
        "reference_size": int(len(reference)),
        "log_transformed_columns": len(encoder.log_columns),
        "evaluation_sets": {name: {"n": int(len(frame)), "attack_rate": float(frame[features.label_column].mean())} for name, frame in evaluation_sets.items()},
        "fit_seconds": bank.fit_seconds,
        "score_seconds": natural_score_seconds,
        "base_rate": base_rate_rows,
        "contamination": contamination_rows,
        "category_recall": category_rows,
        "structure": structure,
        "verification": verification,
    }
    write_results(summary, config)
    _log("finished in " + format(summary["seconds"], ".1f") + " seconds", verbose)
    return summary


def write_results(summary: dict, config: Config) -> Path:
    output = config.output_dir
    output.mkdir(parents=True, exist_ok=True)
    with open(output / "summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
    pd.DataFrame(summary["base_rate"]).to_csv(output / "base_rate.csv", index=False)
    pd.DataFrame(summary["contamination"]).to_csv(output / "contamination.csv", index=False)
    pd.DataFrame(summary["category_recall"]).to_csv(output / "category_recall.csv", index=False)
    pd.DataFrame(summary["structure"]["kmeans_grid"]).to_csv(output / "kmeans_grid.csv", index=False)
    return output
