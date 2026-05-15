from __future__ import annotations

import hashlib

import numpy as np


WHITEBOX_MODELS = {"ds_llama", "ds_qwen", "qwen_2", "llama_3", "qwen_3"}


def transfer_indices(rows: list[dict], dataset: str, transfer: str) -> tuple[np.ndarray, np.ndarray]:
    if dataset == "hs":
        if transfer == "hs1_to_hs2":
            source = [i for i, row in enumerate(rows) if row.get("source") == "hs1"]
            target = [i for i, row in enumerate(rows) if row.get("source") == "hs2"]
        elif transfer == "hs2_to_hs1":
            source = [i for i, row in enumerate(rows) if row.get("source") == "hs2"]
            target = [i for i, row in enumerate(rows) if row.get("source") == "hs1"]
        else:
            raise ValueError(f"unknown HelpSteer transfer: {transfer}")
    elif dataset == "mb":
        if transfer == "whitebox_to_claude":
            target_model = "claude"
        elif transfer == "whitebox_to_gemini":
            target_model = "gemini"
        else:
            raise ValueError(f"unknown MentalBench transfer: {transfer}")
        source = [i for i, row in enumerate(rows) if row.get("model") in WHITEBOX_MODELS]
        target = [i for i, row in enumerate(rows) if row.get("model") == target_model]
    else:
        raise ValueError(f"unknown dataset: {dataset}")
    if not source or not target:
        raise ValueError(f"empty split for {dataset}/{transfer}: source={len(source)} target={len(target)}")
    return np.asarray(source, dtype=np.int64), np.asarray(target, dtype=np.int64)


def deterministic_sample(idx: np.ndarray, size: int | None, seed: int, salt: str) -> np.ndarray:
    idx = np.asarray(idx, dtype=np.int64)
    if size is None or size <= 0 or len(idx) <= size:
        return np.sort(idx)
    salt_int = int.from_bytes(hashlib.sha256(salt.encode()).digest()[:4], "little")
    rng = np.random.default_rng(int(seed) + salt_int)
    return np.sort(rng.choice(idx, size=size, replace=False)).astype(np.int64)


def source_dre_cal_split(
    source_idx: np.ndarray,
    n_dre: int,
    n_cal: int,
    seed: int,
    salt_prefix: str,
) -> tuple[np.ndarray, np.ndarray]:
    if n_dre + n_cal != 2000:
        raise ValueError(f"n_dre + n_cal must be 2000, got {n_dre + n_cal}")
    dre_idx = deterministic_sample(source_idx, n_dre, seed, f"{salt_prefix}/Dscore_Ddre_source")
    remaining = np.setdiff1d(np.asarray(source_idx, dtype=np.int64), dre_idx, assume_unique=False)
    cal_idx = deterministic_sample(remaining, n_cal, seed, f"{salt_prefix}/Dcal")
    if len(dre_idx) != n_dre or len(cal_idx) != n_cal:
        raise ValueError(
            f"requested split cannot be met for {salt_prefix}: "
            f"Dscore/Ddre_source={len(dre_idx)} Dcal={len(cal_idx)} requested={n_dre}/{n_cal}"
        )
    return dre_idx, cal_idx

