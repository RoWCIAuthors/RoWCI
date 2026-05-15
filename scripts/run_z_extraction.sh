#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

: "${INPUT_JSONL:?set INPUT_JSONL}"
: "${Z_JSONL:?set Z_JSONL}"
: "${SCHEMA_JSON:?set SCHEMA_JSON}"
: "${DATASET:?set DATASET}"
: "${AXIS:?set AXIS}"
: "${OUTPUT_JSONL:?set OUTPUT_JSONL}"

ARGS=()
if [[ -n "${SOURCE:-}" ]]; then
  ARGS+=(--source "$SOURCE")
fi
if [[ -n "${PROMPT:-}" && -n "${PROMPT_FILE:-}" ]]; then
  echo "set only one of PROMPT or PROMPT_FILE" >&2
  exit 1
fi
if [[ -n "${PROMPT:-}" ]]; then
  ARGS+=(--prompt "$PROMPT")
fi

resolve_path() {
  case "$1" in
    /*) printf '%s\n' "$1" ;;
    *) printf '%s\n' "$ROOT/$1" ;;
  esac
}

if [[ -n "${PROMPT_FILE:-}" ]]; then
  ARGS+=(--prompt-file "$(resolve_path "$PROMPT_FILE")")
fi

PYTHONDONTWRITEBYTECODE=1 python "$ROOT/src/rowci_deploy/z_extraction.py" \
  --input-jsonl "$(resolve_path "$INPUT_JSONL")" \
  --z-jsonl "$(resolve_path "$Z_JSONL")" \
  --schema "$(resolve_path "$SCHEMA_JSON")" \
  --dataset "$DATASET" \
  --axis "$AXIS" \
  --output-jsonl "$(resolve_path "$OUTPUT_JSONL")" \
  "${ARGS[@]}"
