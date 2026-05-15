from __future__ import annotations

import numpy as np


def z_coordinates(row: dict) -> dict:
    return row.get("z_coordinates") or {}


def z_cell_key(row: dict) -> tuple[tuple[str, str], ...]:
    coords = z_coordinates(row)
    if len(coords) == 1:
        value = next(iter(coords.values()))
        if isinstance(value, str) and "|" in value and "=" in value:
            return tuple(tuple(part.split("=", 1)) for part in sorted(value.split("|")))
    return tuple(sorted((str(key), str(value)) for key, value in coords.items()))


def onehot_z(rows: list[dict]) -> tuple[np.ndarray, list[str]]:
    feature_dicts = []
    for i, row in enumerate(rows):
        feats = {f"{key}={value}": 1.0 for key, value in z_cell_key(row)}
        if not feats:
            raise ValueError(f"missing Z coordinates for row {i}")
        feature_dicts.append(feats)
    names = sorted({name for feats in feature_dicts for name in feats})
    lookup = {name: i for i, name in enumerate(names)}
    x = np.zeros((len(rows), len(names)), dtype=np.float32)
    for i, feats in enumerate(feature_dicts):
        for name, value in feats.items():
            x[i, lookup[name]] = float(value)
    return x, names


def _clip_weights(w: np.ndarray, clip_quantile: float) -> np.ndarray:
    if clip_quantile < 95.0 or clip_quantile > 100.0:
        raise ValueError(f"weight clipping quantile must be in [95, 100], got {clip_quantile}")
    w = np.asarray(w, dtype=np.float64)
    if clip_quantile < 100.0:
        w = np.clip(w, 0.0, np.percentile(w, clip_quantile))
    return w


def classifier_density_ratio(
    x: np.ndarray,
    source_idx: np.ndarray,
    target_idx: np.ndarray,
    apply_idx: np.ndarray,
    model_name: str,
    params: dict,
    seed: int,
    clip_quantile: float,
) -> np.ndarray:
    source_idx = np.asarray(source_idx, dtype=np.int64)
    target_idx = np.asarray(target_idx, dtype=np.int64)
    apply_idx = np.asarray(apply_idx, dtype=np.int64)
    train_idx = np.concatenate([source_idx, target_idx])
    y_domain = np.concatenate([np.zeros(len(source_idx), dtype=np.int64), np.ones(len(target_idx), dtype=np.int64)])
    pi_source = len(source_idx) / max(1, len(train_idx))
    pi_target = len(target_idx) / max(1, len(train_idx))

    if model_name == "rf":
        from sklearn.ensemble import RandomForestClassifier

        model = RandomForestClassifier(
            n_estimators=int(params.get("n_estimators", 200)),
            max_depth=params.get("max_depth", 8),
            min_samples_leaf=int(params.get("min_samples_leaf", 5)),
            max_features=params.get("max_features", "sqrt"),
            class_weight=params.get("class_weight"),
            random_state=int(seed),
            n_jobs=int(params.get("n_jobs", 2)),
        )
    elif model_name == "extra_trees":
        from sklearn.ensemble import ExtraTreesClassifier

        model = ExtraTreesClassifier(
            n_estimators=int(params.get("n_estimators", 300)),
            max_depth=params.get("max_depth", 8),
            min_samples_leaf=int(params.get("min_samples_leaf", 5)),
            max_features=params.get("max_features", "sqrt"),
            class_weight=params.get("class_weight"),
            random_state=int(seed),
            n_jobs=int(params.get("n_jobs", 2)),
        )
    elif model_name == "logistic":
        from sklearn.linear_model import LogisticRegression

        model = LogisticRegression(
            C=float(params.get("C", 1.0)),
            class_weight=params.get("class_weight"),
            max_iter=int(params.get("max_iter", 2000)),
            random_state=int(seed),
        )
    elif model_name == "hgb":
        from sklearn.ensemble import HistGradientBoostingClassifier

        model = HistGradientBoostingClassifier(
            max_iter=int(params.get("max_iter", 100)),
            max_leaf_nodes=int(params.get("max_leaf_nodes", 15)),
            learning_rate=float(params.get("learning_rate", 0.05)),
            l2_regularization=float(params.get("l2_regularization", 0.0)),
            random_state=int(seed),
        )
    else:
        raise ValueError(f"unknown DRE classifier: {model_name}")

    model.fit(x[train_idx], y_domain)
    eta = np.asarray(model.predict_proba(x[apply_idx])[:, 1], dtype=np.float64)
    eps = float(params.get("eta_floor", 1e-6))
    eta = np.clip(eta, eps, 1.0 - eps)
    weights = eta / (1.0 - eta) * (pi_source / max(pi_target, eps))
    return _clip_weights(weights, clip_quantile)
