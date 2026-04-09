#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CFG="${1:-$ROOT/configs/default.yaml}"
REPORT_PATH="${AUTORESEARCH_REPORT_PATH:-$ROOT/.autoresearch/bench.json}"
WARMUP="${AUTORESEARCH_BENCH_WARMUP:-2}"
LIMIT="${AUTORESEARCH_BENCH_LIMIT:-0}"

mkdir -p "$(dirname "$REPORT_PATH")"
cd "$ROOT"

args=(
  python3 infer.py
  --config "$CFG"
  --benchmark
  --metrics
  --bench_warmup "$WARMUP"
  --bench_limit "$LIMIT"
  --bench_report_json "$REPORT_PATH"
)

if [[ -n "${AUTORESEARCH_BENCH_IMAGES_DIR:-}" ]]; then
  args+=(--bench_images_dir "$AUTORESEARCH_BENCH_IMAGES_DIR")
fi

if [[ -n "${AUTORESEARCH_BENCH_MASKS_DIR:-}" ]]; then
  args+=(--bench_masks_dir "$AUTORESEARCH_BENCH_MASKS_DIR")
fi

if [[ -n "${AUTORESEARCH_BENCH_SIZE_FILTER:-}" ]]; then
  args+=(--bench_size_filter "$AUTORESEARCH_BENCH_SIZE_FILTER")
fi

exec "${args[@]}"
