from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Callable

try:
    from .io import load_jsonl, write_jsonl
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from rowci_deploy.io import load_jsonl, write_jsonl


EXTRACTION_PROMPT = ""
WHITEBOX_MODELS = {"ds_llama", "ds_qwen", "qwen_2", "llama_3", "qwen_3"}


def load_schema(schema_path: Path) -> dict[str, list[str]]:
    with schema_path.open() as f:
        schema = json.load(f)
    return schema["coordinates"]


def label_for(row: dict, axis: str) -> float:
    key = f"y_{axis}"
    if key in row:
        return float(row[key])
    if "y" in row:
        return float(row["y"])
    raise KeyError(f"missing label field {key}")


def source_for(row: dict, dataset: str, source: str | None) -> str | None:
    if dataset == "hs":
        if source is None:
            raise ValueError("HS extraction requires --source hs1 or --source hs2")
        return source
    if dataset == "mb":
        model = str(row.get("model"))
        return "weak" if model in WHITEBOX_MODELS else model
    raise ValueError(f"unknown dataset: {dataset}")


def item_id_for(row: dict, dataset: str, axis: str, source: str | None, row_index: int) -> str:
    idx = row.get("idx", row_index)
    if dataset == "hs":
        return f"{dataset}_{axis}_{source}_{int(idx):04d}"
    return str(idx)


def state_key(z_coordinates: dict[str, str]) -> str:
    def key_order(name: str) -> tuple[int, str]:
        if name.startswith("Z") and name[1:].isdigit():
            return int(name[1:]), name
        return 10_000, name

    return "|".join(f"{key}={z_coordinates[key]}" for key in sorted(z_coordinates, key=key_order))


def validate_z(z_coordinates: dict[str, str], schema: dict[str, list[str]]) -> None:
    if set(z_coordinates) != set(schema):
        raise ValueError(f"Z coordinate keys must be {sorted(schema)}, got {sorted(z_coordinates)}")
    for key, value in z_coordinates.items():
        if value not in schema[key]:
            raise ValueError(f"invalid {key}={value}; expected one of {schema[key]}")


def extraction_input(row: dict) -> dict:
    blocked = {"y", "raw_score", "base_score", "z", "z_coordinates", "coordinate_state_key"}
    return {key: value for key, value in row.items() if key not in blocked and not key.startswith("y_")}


def extract_z_coordinates(
    row: dict,
    schema: dict[str, list[str]],
    extractor: Callable[[dict, dict[str, list[str]], str], dict[str, str]] | None,
    prompt: str,
) -> dict[str, str]:
    existing = row.get("z_coordinates")
    if isinstance(existing, dict):
        validate_z(existing, schema)
        return {str(key): str(value) for key, value in existing.items()}
    if extractor is None:
        raise RuntimeError("Z extraction backend is not configured")
    z_coordinates = extractor(extraction_input(row), schema, prompt)
    validate_z(z_coordinates, schema)
    return z_coordinates


def build_extracted_rows(
    rows: list[dict],
    dataset: str,
    axis: str,
    schema: dict[str, list[str]],
    source: str | None = None,
    extractor: Callable[[dict, dict[str, list[str]], str], dict[str, str]] | None = None,
    prompt: str = EXTRACTION_PROMPT,
) -> list[dict]:
    out = []
    for row_index, row in enumerate(rows):
        z_coordinates = extract_z_coordinates(row, schema, extractor, prompt)
        key = state_key(z_coordinates)
        out.append(
            {
                "item_id": item_id_for(row, dataset, axis, source, row_index),
                "idx": row.get("idx", row_index),
                "dataset": dataset,
                "axis": axis,
                "source": source_for(row, dataset, source),
                "model": row.get("model"),
                "y": label_for(row, axis),
                "z": {"state_key": key},
                "z_coordinates": z_coordinates,
                "coordinate_state_key": key,
                "row_index": row_index,
            }
        )
    return out


def attach_z_rows(rows: list[dict], z_rows: list[dict]) -> list[dict]:
    def z_payload(z_row: dict) -> dict:
        if "z_coordinates" in z_row:
            return z_row["z_coordinates"]
        if "coordinates" in z_row:
            return z_row["coordinates"]
        return z_row

    if z_rows and all("idx" in z_row for z_row in z_rows):
        by_idx = {str(row.get("idx")): row for row in rows}
        merged = []
        for z_row in z_rows:
            key = str(z_row["idx"])
            if key not in by_idx:
                raise ValueError(f"missing source row for idx={key}")
            current = dict(by_idx[key])
            current["z_coordinates"] = z_payload(z_row)
            merged.append(current)
        return merged

    if len(rows) != len(z_rows):
        raise ValueError(f"row count mismatch without idx join key: data={len(rows)} z={len(z_rows)}")
    merged = []
    for row, z_row in zip(rows, z_rows):
        current = dict(row)
        current["z_coordinates"] = z_payload(z_row)
        merged.append(current)
    return merged


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build extracted_Z-format JSONL from a data JSONL file.")
    parser.add_argument("--input-jsonl", type=Path, required=True)
    parser.add_argument("--z-jsonl", type=Path, default=None)
    parser.add_argument("--prompt", type=str, default=None)
    parser.add_argument("--prompt-file", type=Path, default=None)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--schema", type=Path, required=True)
    parser.add_argument("--dataset", choices=["hs", "mb"], required=True)
    parser.add_argument("--axis", choices=["correctness", "helpfulness", "guidance"], required=True)
    parser.add_argument("--source", choices=["hs1", "hs2"], default=None)
    return parser.parse_args()


def load_prompt(args: argparse.Namespace) -> str:
    if args.prompt is not None and args.prompt_file is not None:
        raise ValueError("use only one of --prompt or --prompt-file")
    if args.prompt_file is not None:
        return args.prompt_file.read_text()
    if args.prompt is not None:
        return args.prompt
    return EXTRACTION_PROMPT


def main() -> None:
    args = parse_args()
    schema = load_schema(args.schema)
    prompt = load_prompt(args)
    rows = load_jsonl(args.input_jsonl)
    if args.z_jsonl:
        rows = attach_z_rows(rows, load_jsonl(args.z_jsonl))
    extracted = build_extracted_rows(
        rows=rows,
        dataset=args.dataset,
        axis=args.axis,
        schema=schema,
        source=args.source,
        prompt=prompt,
    )
    write_jsonl(args.output_jsonl, extracted)


if __name__ == "__main__":
    main()
