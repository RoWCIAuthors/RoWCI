from __future__ import annotations

import math

import numpy as np


def score_values_for(dataset: str) -> list[int]:
    if dataset == "hs":
        return [0, 1, 2, 3, 4]
    if dataset == "mb":
        return [1, 2, 3, 4, 5]
    raise ValueError(f"unknown dataset: {dataset}")


def clamp_for(dataset: str) -> tuple[float, float]:
    values = score_values_for(dataset)
    return float(min(values)), float(max(values))


def standard_quantile(values: np.ndarray, alpha: float) -> float:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if len(finite) == 0:
        return math.nan
    k = min(len(finite), int(math.ceil((1.0 - alpha) * (len(finite) + 1))))
    return float(np.sort(finite)[k - 1])


def weighted_quantile(values: np.ndarray, weights: np.ndarray, alpha: float) -> float:
    values = np.asarray(values, dtype=np.float64)
    weights = np.maximum(np.asarray(weights, dtype=np.float64), 0.0)
    mask = np.isfinite(values) & np.isfinite(weights)
    values = values[mask]
    weights = weights[mask]
    if len(values) == 0:
        return math.nan
    if float(weights.sum()) <= 0:
        return standard_quantile(values, alpha)
    order = np.argsort(values)
    sorted_values = values[order]
    cumsum = np.cumsum(weights[order])
    threshold = (1.0 - alpha) * (float(cumsum[-1]) + 1.0)
    if threshold > cumsum[-1]:
        return math.inf
    idx = int(np.searchsorted(cumsum, threshold, side="left"))
    return float(sorted_values[max(0, min(idx, len(sorted_values) - 1))])


def interval_metrics(
    y: np.ndarray,
    score: np.ndarray,
    qhat: float,
    clamp: tuple[float, float],
    weights: np.ndarray | None = None,
) -> dict[str, float]:
    y = np.asarray(y, dtype=np.float64)
    score = np.asarray(score, dtype=np.float64)
    lower = np.clip(score - qhat, clamp[0], clamp[1])
    upper = np.clip(score + qhat, clamp[0], clamp[1])
    mask = np.isfinite(y) & np.isfinite(lower) & np.isfinite(upper)
    covered = (y[mask] >= lower[mask]) & (y[mask] <= upper[mask])
    lengths = upper[mask] - lower[mask]
    raw_size = float(np.mean(lengths)) if len(lengths) else math.nan
    raw_size_std = float(np.std(lengths)) if len(lengths) else math.nan
    out = {
        "coverage": float(np.mean(covered)) if len(covered) else math.nan,
        "size": raw_size / 5.0,
        "size_std": raw_size_std / 5.0,
        "raw_size": raw_size,
        "raw_size_std": raw_size_std,
        "q_hat": float(qhat),
    }
    if weights is not None:
        w = np.asarray(weights, dtype=np.float64)
        w = w[np.isfinite(w)]
        ess = float(w.sum() ** 2 / np.sum(w * w)) if len(w) and np.sum(w * w) > 0 else 0.0
        out["ess"] = ess
        out["ess_ratio"] = ess / max(1, len(w))
    return out
