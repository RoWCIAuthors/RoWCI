from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path

import numpy as np

from .conformal import clamp_for, interval_metrics, score_values_for, standard_quantile, weighted_quantile
from .features import classifier_density_ratio, onehot_z
from .io import load_jsonl, write_csv, write_jsonl
from .scoring import fit_xgb_classifier_expected, predict_expected
from .splits import deterministic_sample, source_dre_cal_split, transfer_indices


TASKS = {
    "hs": {"axes": ["correctness", "helpfulness"], "transfers": ["hs1_to_hs2", "hs2_to_hs1"]},
    "mb": {"axes": ["guidance"], "transfers": ["whitebox_to_claude", "whitebox_to_gemini"]},
}


def load_rows(data_root: Path, dataset: str, axis: str, filename: str) -> list[dict]:
    path = data_root / dataset / axis / filename
    if not path.exists():
        raise FileNotFoundError(path)
    rows = load_jsonl(path)
    if filename == "extracted_z.jsonl":
        validate_extracted_z(rows, data_root / "schema" / f"{axis}.json")
    return rows


def validate_extracted_z(rows: list[dict], schema_path: Path) -> None:
    if not schema_path.exists():
        raise FileNotFoundError(schema_path)
    with schema_path.open() as f:
        schema = json.load(f)["coordinates"]
    expected_keys = set(schema)
    for i, row in enumerate(rows):
        coords = row.get("z_coordinates")
        if not isinstance(coords, dict):
            raise ValueError(f"missing z_coordinates for row {i}")
        if set(coords) != expected_keys:
            raise ValueError(f"invalid Z keys for row {i}: expected {sorted(expected_keys)}, got {sorted(coords)}")
        for key, value in coords.items():
            if value not in schema[key]:
                raise ValueError(f"invalid Z value for row {i}: {key}={value}")
        z = row.get("z")
        if z is not None and set(z) != {"state_key"}:
            raise ValueError(f"z field must contain only state_key for row {i}")


def iter_tasks(config: dict) -> list[tuple[str, str, str]]:
    requested = config.get("tasks")
    if requested:
        return [(t["dataset"], t["axis"], t["transfer"]) for t in requested]
    tasks = []
    for dataset, spec in TASKS.items():
        for axis in spec["axes"]:
            for transfer in spec["transfers"]:
                tasks.append((dataset, axis, transfer))
    return tasks


def _target_size(config: dict, dataset: str) -> int:
    sizes = config.get("target_size", {})
    defaults = {"hs": 1000, "mb": 500}
    return int(sizes.get(dataset, sizes.get("default", defaults[dataset])))


def _base_score(rows: list[dict]) -> np.ndarray:
    values = []
    for i, row in enumerate(rows):
        score = row.get("base_score", row.get("raw_score"))
        if score is None:
            raise KeyError(f"missing base_score/raw_score for row {i}")
        values.append(float(score))
    return np.asarray(values, dtype=np.float64)


def _wci_embeddings_path(config: dict, dataset: str, axis: str) -> Path:
    paths = config.get("wci", {}).get("embeddings_path")
    if isinstance(paths, dict):
        path = paths.get(dataset, {}).get(axis) if isinstance(paths.get(dataset), dict) else paths.get(dataset)
    else:
        path = paths
    if not path:
        raise ValueError("WCI requires wci.embeddings_path with cached Qwen3-Embedding-8B embeddings")
    return Path(path)


def _y(rows: list[dict]) -> np.ndarray:
    return np.asarray([float(row["y"]) for row in rows], dtype=np.float64)


def _weight_stats(weights: np.ndarray) -> dict[str, float]:
    return {
        "weight_mean": float(np.mean(weights)),
        "weight_std": float(np.std(weights)),
        "weight_p95": float(np.percentile(weights, 95)),
        "weight_p99": float(np.percentile(weights, 99)),
        "weight_max": float(np.max(weights)),
    }


def _fit_rowci_scores(
    rows: list[dict],
    dataset: str,
    dre_idx: np.ndarray,
    seed: int,
    xgb_params: dict,
) -> tuple[np.ndarray, np.ndarray]:
    x_z, _ = onehot_z(rows)
    y = _y(rows)
    values = score_values_for(dataset)
    model = fit_xgb_classifier_expected(x_z[dre_idx], y[dre_idx], values, seed, xgb_params)
    scores = predict_expected(model, x_z, values)
    lo, hi = clamp_for(dataset)
    return np.clip(scores, lo, hi), x_z


def run_one(rows: list[dict], dataset: str, axis: str, transfer: str, seed: int, method: str, config: dict) -> dict:
    alpha = float(config.get("alpha", 0.1))
    n_dre = int(config.get("n_dre", 1000))
    n_cal = int(config.get("n_cal", 1000))
    target_size = _target_size(config, dataset)
    clip_quantile = float(config.get("weight_clip_quantile", 95.0))

    source_idx, target_pool_idx = transfer_indices(rows, dataset, transfer)
    salt = f"{dataset}/{axis}/{transfer}"
    dre_idx, cal_idx = source_dre_cal_split(source_idx, n_dre, n_cal, seed, salt)
    target_idx = deterministic_sample(target_pool_idx, target_size, seed, f"{salt}/Dc")
    if len(target_idx) != target_size:
        raise ValueError(f"target split too small for {salt}: got {len(target_idx)}, requested {target_size}")

    y = _y(rows)
    clamp = clamp_for(dataset)

    if method == "rowci":
        scores, x_z = _fit_rowci_scores(rows, dataset, dre_idx, seed, config.get("xgboost", {}))
        errors_cal = np.abs(scores[cal_idx] - y[cal_idx])
        dre_cfg = config.get("dre", {})
        weights = classifier_density_ratio(
            x_z,
            dre_idx,
            target_idx,
            cal_idx,
            model_name=dre_cfg.get("model", "rf"),
            params=dre_cfg.get("params", {}),
            seed=seed,
            clip_quantile=clip_quantile,
        )
        qhat = weighted_quantile(errors_cal, weights, alpha)
        metrics = interval_metrics(y[target_idx], scores[target_idx], qhat, clamp, weights)
        score_name = "xgboost_z_classifier_expected"
        dre_name = f"z_classifier_{dre_cfg.get('model', 'rf')}"
    elif method == "sci":
        scores = _base_score(rows)
        errors_cal = np.abs(scores[cal_idx] - y[cal_idx])
        qhat = standard_quantile(errors_cal, alpha)
        weights = np.ones(len(cal_idx), dtype=np.float64)
        metrics = interval_metrics(y[target_idx], scores[target_idx], qhat, clamp, weights)
        score_name = "base_score"
        dre_name = "none"
    elif method == "wci":
        embeddings_path = _wci_embeddings_path(config, dataset, axis)
        x = np.load(embeddings_path)
        if len(x) != len(rows):
            raise ValueError(f"embedding row count mismatch: {len(x)} != {len(rows)}")
        scores = _base_score(rows)
        errors_cal = np.abs(scores[cal_idx] - y[cal_idx])
        weights = classifier_density_ratio(
            x,
            dre_idx,
            target_idx,
            cal_idx,
            model_name="rf",
            params=config.get("wci", {}).get("rf_params", {}),
            seed=seed,
            clip_quantile=clip_quantile,
        )
        qhat = weighted_quantile(errors_cal, weights, alpha)
        metrics = interval_metrics(y[target_idx], scores[target_idx], qhat, clamp, weights)
        score_name = "base_score"
        dre_name = "embedding_rf"
    else:
        raise ValueError(f"unknown method: {method}")

    return {
        "dataset": dataset,
        "axis": axis,
        "transfer": transfer,
        "seed": int(seed),
        "method": method.upper() if method != "rowci" else "RoWCI",
        "alpha": alpha,
        "n_dre": n_dre,
        "n_cal": n_cal,
        "n_target": int(len(target_idx)),
        "score_model": score_name,
        "dre": dre_name,
        "weight_clip_quantile": clip_quantile if method != "sci" else None,
        **_weight_stats(weights),
        **metrics,
    }


def summarize(rows: list[dict]) -> list[dict]:
    groups = defaultdict(list)
    for row in rows:
        groups[(row["dataset"], row["axis"], row["transfer"], row["method"])].append(row)
    summary = []
    metrics = [
        "coverage",
        "size",
        "size_std",
        "raw_size",
        "raw_size_std",
        "q_hat",
        "ess_ratio",
        "weight_mean",
        "weight_std",
        "weight_p99",
        "weight_max",
    ]
    for key, items in sorted(groups.items()):
        dataset, axis, transfer, method = key
        out = {
            "dataset": dataset,
            "axis": axis,
            "transfer": transfer,
            "method": method,
            "n_repeats": len(items),
            "n_dre": items[0]["n_dre"],
            "n_cal": items[0]["n_cal"],
            "n_target": items[0]["n_target"],
        }
        for metric in metrics:
            values = np.asarray([float(row[metric]) for row in items if row.get(metric) is not None], dtype=np.float64)
            out[f"{metric}_mean"] = float(np.mean(values)) if len(values) else None
            out[f"{metric}_var"] = float(np.var(values)) if len(values) else None
            out[f"{metric}_std"] = float(np.std(values)) if len(values) else None
            out[f"{metric}_min"] = float(np.min(values)) if len(values) else None
            out[f"{metric}_max"] = float(np.max(values)) if len(values) else None
        out["coverage_ok"] = bool(out["coverage_mean"] is not None and out["coverage_mean"] >= 0.90)
        summary.append(out)
    return summary


def run(config: dict, method: str, out_dir: Path) -> list[dict]:
    if method == "rowci":
        data_root = Path(config.get("data_root", "data/extracted_Z"))
        filename = "extracted_z.jsonl"
    else:
        base_score_root = config.get("base_score_root")
        if not base_score_root:
            raise ValueError(f"{method.upper()} requires base_score_root with Z-independent cached base scores")
        data_root = Path(base_score_root)
        filename = "base_scores.jsonl"
    seeds = [int(seed) for seed in config.get("seeds", list(range(20)))]
    all_rows = []
    cache: dict[tuple[str, str], list[dict]] = {}
    for dataset, axis, transfer in iter_tasks(config):
        rows = cache.setdefault((dataset, axis), load_rows(data_root, dataset, axis, filename))
        for seed in seeds:
            all_rows.append(run_one(rows, dataset, axis, transfer, seed, method, config))
    out_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(out_dir / "repeats.jsonl", all_rows)
    write_csv(out_dir / "repeats.csv", all_rows)
    write_csv(out_dir / "summary.csv", summarize(all_rows))
    return all_rows
