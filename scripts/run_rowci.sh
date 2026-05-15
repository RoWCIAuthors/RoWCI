#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHONDONTWRITEBYTECODE=1 python "$ROOT/main.py" --method rowci --config "$ROOT/configs/rowci_default.yaml" --out-dir "$ROOT/outputs/rowci"
