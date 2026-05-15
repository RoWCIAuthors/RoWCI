#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
: "${CONFIG:?set CONFIG to a config with base_score_root}"
OUT_DIR="${OUT_DIR:-$ROOT/outputs/sci}"
case "$CONFIG" in
  /*) CONFIG_PATH="$CONFIG" ;;
  *) CONFIG_PATH="$ROOT/$CONFIG" ;;
esac
PYTHONDONTWRITEBYTECODE=1 python "$ROOT/main.py" --method sci --config "$CONFIG_PATH" --out-dir "$OUT_DIR"
