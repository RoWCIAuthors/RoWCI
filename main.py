from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from rowci_deploy.runner import run, summarize


def resolve_config_paths(value, base: Path):
    if isinstance(value, dict):
        return {key: resolve_config_paths(item, base) for key, item in value.items()}
    if isinstance(value, list):
        return [resolve_config_paths(item, base) for item in value]
    if isinstance(value, str):
        path = Path(value)
        return str((base / path).resolve()) if not path.is_absolute() else str(path)
    return value


def load_config(path: Path) -> dict:
    with path.open() as f:
        config = yaml.safe_load(f) or {}
    for key in ("data_root", "base_score_root"):
        if config.get(key):
            config[key] = resolve_config_paths(config[key], path.parent)
    if config.get("wci", {}).get("embeddings_path"):
        config["wci"]["embeddings_path"] = resolve_config_paths(config["wci"]["embeddings_path"], path.parent)
    return config


def main() -> None:
    parser = argparse.ArgumentParser(description="Run SCI, WCI, or RoWCI. Default method is RoWCI.")
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "rowci_default.yaml")
    parser.add_argument("--method", choices=["rowci", "sci", "wci"], default="rowci")
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--summary-json", type=Path, default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    out_dir = args.out_dir or ROOT / "outputs" / args.method
    rows = run(config, args.method, out_dir)
    summary = summarize(rows)
    if args.summary_json:
        args.summary_json.parent.mkdir(parents=True, exist_ok=True)
        with args.summary_json.open("w") as f:
            json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
