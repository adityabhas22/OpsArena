#!/usr/bin/env bash

set -euo pipefail

MODEL="${1:-Qwen/Qwen3-4B-Instruct-2507}"
SEEDS="${2:-1,2,3,4,5,6,7,8,9,10,11,12}"
OUT_DIR="${3:-artifacts/benchmarks/refund}"

mkdir -p "$OUT_DIR"

run_one() {
  local label="$1"
  shift

  echo "[$(date --iso-8601=seconds)] starting ${label}"
  python scripts/benchmark_refund_models.py \
    --model "$MODEL" \
    --seeds "$SEEDS" \
    --include-oracle \
    --output-json "$OUT_DIR/${label}.json" \
    "$@" \
    | tee "$OUT_DIR/${label}.log"
  echo "[$(date --iso-8601=seconds)] finished ${label}"
}

run_one base
run_one checkpoint_150 --adapter artifacts/refund-grpo/checkpoint-150
run_one final_adapter --adapter artifacts/refund-grpo

