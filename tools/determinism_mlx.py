"""Measure bit-exact output stability across repeated temp=0 runs on MLX.

Produces a matrix: (config x prompt x run_index) -> token sequence, then
reports the fraction of runs that are bit-identical per (config, prompt).
"""
import json
from pathlib import Path

import mlx.core as mx
from mlx_lm.sample_utils import make_sampler
from mlx_lm import stream_generate as baseline_stream

from dflash.model_mlx import load, load_draft, stream_generate as dflash_stream


MODEL = "mlx-community/Qwen3.5-27B-4bit"
DRAFT = "z-lab/Qwen3.5-27B-DFlash"
BLOCK = 16
RUNS_PER_PROMPT = 10
MAX_NEW = 128

PROMPTS = [
    "What is 2+2? Think step by step, then answer.",
    "Write a Python function that reverses a string.",
    "Explain RSA encryption in one paragraph.",
]


def main():
    print("Loading target + drafter (this takes a minute)...")
    model, tok = load(MODEL)
    draft = load_draft(DRAFT, sliding_window_size=4096)
    sampler = make_sampler(temp=0.0)

    def run_baseline(prompt):
        return [r.token for r in baseline_stream(model, tok, prompt, MAX_NEW, sampler=sampler)]

    def run_dflash(prompt):
        return [t for r in dflash_stream(model, draft, tok, prompt, BLOCK, MAX_NEW, sampler=sampler) for t in r.tokens]

    results = {"peak_memory_gb": {}}
    for config_name, runner in [("baseline_mlx_lm", run_baseline), ("dflash", run_dflash)]:
        mx.reset_peak_memory()
        results[config_name] = {}
        for prompt in PROMPTS:
            outputs = [tuple(runner(prompt)) for _ in range(RUNS_PER_PROMPT)]
            unique = set(outputs)
            results[config_name][prompt[:40]] = {
                "runs": len(outputs),
                "unique_outputs": len(unique),
                "bit_exact_rate": outputs.count(outputs[0]) / len(outputs),
                "token_lengths": [len(o) for o in outputs],
            }
            print(f"  [{config_name}] {prompt[:40]!r}: {len(unique)} unique / {len(outputs)} runs")
        results["peak_memory_gb"][config_name] = mx.get_peak_memory() / 1e9
        print(f"  [{config_name}] peak memory: {results['peak_memory_gb'][config_name]:.2f} GB")

    out = Path("~/github/dflash/determinism_mlx_results.json").expanduser()
    out.write_text(json.dumps(results, indent=2))
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()
