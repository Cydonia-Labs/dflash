"""Measure bit-exact output stability across repeated temp=0 requests to SGLang.

Companion to tools/determinism_mlx.py — runs the same 3 prompts × 10 runs test
but against an already-running SGLang server on mcai, via the OpenAI-compatible
/v1/chat/completions endpoint.

Usage:
    python tools/determinism_mcai.py <config_label>

Where <config_label> is an arbitrary string identifying the current server
config (e.g., "baseline_sglang_triton", "dflash_sglang_triton"). Results are
merged into /tmp/determinism_mcai_results.json so repeated invocations with
different labels accumulate into one file for comparison.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import requests

URL = "http://127.0.0.1:30000"
PROMPTS = [
    "What is 2+2? Think step by step, then answer.",
    "Write a Python function that reverses a string.",
    "Explain RSA encryption in one paragraph.",
]
RUNS_PER_PROMPT = 10
MAX_NEW_TOKENS = 128
RESULTS_FILE = Path("/tmp/determinism_mcai_results.json")


def run_determinism(label: str) -> dict:
    """Send each prompt RUNS_PER_PROMPT times at temp=0 and record outputs.

    Args:
        label: Arbitrary config identifier for this run (used as top-level key
            in the results dict).

    Returns:
        Dict keyed by label → prompt_prefix → {runs, unique_outputs, bit_exact_rate}.
    """
    out: dict = {label: {}}
    for prompt in PROMPTS:
        outputs: list[str] = []
        for i in range(RUNS_PER_PROMPT):
            r = requests.post(
                URL + "/v1/chat/completions",
                json={
                    "model": "Qwen/Qwen3.5-27B",
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": MAX_NEW_TOKENS,
                    # temp=0 forces greedy; top_p/top_k are ignored but set for clarity
                    "temperature": 0.0,
                    "top_p": 1.0,
                    "top_k": 1,
                },
                timeout=300,
            )
            r.raise_for_status()
            content = r.json()["choices"][0]["message"]["content"]
            outputs.append(content)
            print(f"  [{label}] run {i + 1}/{RUNS_PER_PROMPT} for {prompt[:30]!r}: {len(content)} chars")

        unique = set(outputs)
        # bit_exact_rate = fraction of runs that matched the first run
        bit_exact_rate = outputs.count(outputs[0]) / len(outputs)
        out[label][prompt[:40]] = {
            "runs": len(outputs),
            "unique_outputs": len(unique),
            "bit_exact_rate": bit_exact_rate,
            "first_32chars_first_run": outputs[0][:32],
            # Save the variants if non-deterministic for post-hoc inspection
            "variants_preview": [o[:64] for o in list(unique)[:3]] if len(unique) > 1 else None,
        }
        print(f"[{label}] {prompt[:40]!r}: {len(unique)} unique / {len(outputs)} runs, rate={bit_exact_rate}")

    return out


def main() -> None:
    if len(sys.argv) != 2:
        print("Usage: python determinism_mcai.py <config_label>", file=sys.stderr)
        sys.exit(1)
    label = sys.argv[1]
    new = run_determinism(label)
    # Merge with existing results if the file exists, so baseline + dflash runs
    # accumulate into a single comparison file.
    if RESULTS_FILE.exists():
        existing = json.loads(RESULTS_FILE.read_text())
        existing.update(new)
        new = existing
    RESULTS_FILE.write_text(json.dumps(new, indent=2))
    print(f"\nsaved to {RESULTS_FILE}")


if __name__ == "__main__":
    main()
