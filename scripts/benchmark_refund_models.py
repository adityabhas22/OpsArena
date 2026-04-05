#!/usr/bin/env python3
"""Compare base vs fine-tuned refund policies on the same seeds (fair benchmark).

Requires the same dependencies as GRPO training: ``pip install -e ".[train]"``.

Example::

    # Base instruct model only
    python scripts/benchmark_refund_models.py --model Qwen/Qwen3-4B-Instruct-2507 --seeds 1,2,3,4

    # LoRA / merged checkpoint from training
    python scripts/benchmark_refund_models.py \\
        --model Qwen/Qwen3-4B-Instruct-2507 \\
        --adapter artifacts/refund-grpo \\
        --seeds 1,2,3,4

Oracle scores use ``baselines.oracle.run_oracle`` on the full env (ceiling reference).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _parse_seeds(s: str) -> list[int]:
    parts = [p.strip() for p in s.split(",") if p.strip()]
    return [int(p) for p in parts]


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark refund policies (base vs LoRA) on shared seeds.")
    parser.add_argument(
        "--model",
        required=True,
        help="Base Hugging Face model id (must match training, e.g. Qwen/Qwen3-4B-Instruct-2507).",
    )
    parser.add_argument(
        "--adapter",
        default=None,
        help="Path to LoRA output dir from train_refund_grpo.py (merged into base for eval).",
    )
    parser.add_argument("--seeds", default="1,2,3,4,5,6,7,8", help="Comma-separated scenario seeds.")
    parser.add_argument("--max-turns", type=int, default=80)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.0, help="0 = greedy decoding.")
    parser.add_argument("--torch-dtype", default="bfloat16", help="torch dtype name, e.g. bfloat16 or float16.")
    parser.add_argument(
        "--include-oracle",
        action="store_true",
        help="Also run baselines.oracle per seed (privileged upper bound).",
    )
    parser.add_argument(
        "--output-json",
        default=None,
        help="Write full result dict to this path.",
    )
    parser.add_argument(
        "--save-transcripts",
        action="store_true",
        help="Include full chat transcripts in JSON (large).",
    )
    args = parser.parse_args()

    try:
        from opsarena.training.refund_eval import load_policy_model, run_refund_episode_local
        from baselines.oracle import run_oracle
    except ImportError as exc:
        raise SystemExit(
            "Missing dependencies. Install with: pip install -e '.[train]' "
            f"and ensure PyTorch is available.\n({exc})"
        ) from exc

    seeds = _parse_seeds(args.seeds)
    model, tokenizer = load_policy_model(
        args.model,
        adapter_path=args.adapter,
        torch_dtype=args.torch_dtype,
    )

    rows: list[dict[str, object]] = []
    for seed in seeds:
        result = run_refund_episode_local(
            model,
            tokenizer,
            seed=seed,
            max_turns=args.max_turns,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
        )
        row: dict[str, object] = {
            "seed": seed,
            "benchmark_score": result.benchmark_score,
            "done": result.done,
            "turns": result.turns,
            "parse_failures": result.parse_failures,
            "invalid_actions": result.invalid_actions,
        }
        if args.save_transcripts:
            row["messages"] = result.messages
        rows.append(row)
        print(
            f"seed={seed}  score={result.benchmark_score:.4f}  done={result.done}  "
            f"turns={result.turns}  parse_failures={result.parse_failures}  invalid_actions={result.invalid_actions}"
        )

    scores = [float(r["benchmark_score"]) for r in rows]
    summary = {
        "model": args.model,
        "adapter": args.adapter,
        "seeds": seeds,
        "mean_benchmark_score": sum(scores) / len(scores) if scores else 0.0,
        "min_benchmark_score": min(scores) if scores else 0.0,
        "max_benchmark_score": max(scores) if scores else 0.0,
        "episodes": rows,
    }

    if args.include_oracle:
        oracle_scores: list[float] = []
        for seed in seeds:
            st = run_oracle("refund_exception", seed=seed)
            bs = float(st.get("benchmark_score", 0.0))
            oracle_scores.append(bs)
            print(f"oracle seed={seed}  benchmark_score={bs:.4f}")
        summary["oracle_mean_benchmark_score"] = (
            sum(oracle_scores) / len(oracle_scores) if oracle_scores else 0.0
        )
        summary["oracle_scores"] = list(zip(seeds, oracle_scores, strict=True))

    print("---")
    print(f"mean benchmark_score: {summary['mean_benchmark_score']:.4f}")
    if args.include_oracle:
        print(f"oracle mean benchmark_score: {summary['oracle_mean_benchmark_score']:.4f}")

    if args.output_json:
        out_path = Path(args.output_json)
        out_path.write_text(json.dumps(summary, indent=2, default=str))
        print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
