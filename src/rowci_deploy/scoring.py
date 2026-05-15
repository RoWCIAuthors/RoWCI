from __future__ import annotations

import numpy as np


def fit_xgb_classifier_expected(
    x_train: np.ndarray,
    y_train: np.ndarray,
    score_values: list[int],
    seed: int,
    params: dict,
):
    from xgboost import XGBClassifier

    values = np.asarray(score_values, dtype=np.float64)
    y = np.asarray(y_train, dtype=np.float64)
    if not np.all(np.isfinite(y)):
        raise ValueError("non-finite training label for XGBoost score model")
    if np.any(~np.isin(y, values)):
        raise ValueError(f"training labels must be in {score_values}")
    labels = np.asarray([int(np.where(values == float(label))[0][0]) for label in y], dtype=np.int64)
    sample_weight = np.ones(len(labels), dtype=np.float64)
    missing = [klass for klass in range(len(score_values)) if klass not in set(labels.tolist())]
    if missing:
        x_train = np.concatenate([x_train, np.zeros((len(missing), x_train.shape[1]), dtype=x_train.dtype)], axis=0)
        labels = np.concatenate([labels, np.asarray(missing, dtype=np.int64)])
        sample_weight = np.concatenate([sample_weight, np.zeros(len(missing), dtype=np.float64)])
    model = XGBClassifier(
        objective="multi:softprob",
        num_class=len(score_values),
        n_estimators=int(params.get("n_estimators", 100)),
        max_depth=int(params.get("max_depth", 2)),
        learning_rate=float(params.get("learning_rate", 0.05)),
        subsample=float(params.get("subsample", 1.0)),
        colsample_bytree=float(params.get("colsample_bytree", 1.0)),
        min_child_weight=float(params.get("min_child_weight", 1.0)),
        reg_lambda=float(params.get("reg_lambda", 1.0)),
        random_state=int(seed),
        n_jobs=int(params.get("n_jobs", 2)),
        verbosity=0,
        eval_metric="mlogloss",
    )
    model.fit(x_train, labels, sample_weight=sample_weight)
    return model


def predict_expected(model, x: np.ndarray, score_values: list[int]) -> np.ndarray:
    probs = np.asarray(model.predict_proba(x), dtype=np.float64)
    return probs @ np.asarray(score_values, dtype=np.float64)


def score_with_llm_as_judge(*_args, **_kwargs):
    raise RuntimeError(
        "LLM-as-a-judge scoring requires a local Qwen-3-30B model checkpoint. "
        "Generate or cache raw scores before running conformal evaluation."
    )
