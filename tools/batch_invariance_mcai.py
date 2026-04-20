"""Measure batch-variance-driven divergence in greedy decoding on SGLang.

Mirrors mcbook's Test B methodology for mcai (CUDA + triton + SGLang). For each
target prompt, compares the target's solo output vs. the target's output when
run concurrently with varying batchmate sets. Under batch-invariance, all
comparisons should be bit-exact at temp=0. Under batch-variance (reduction-order
drift), concurrent runs may diverge — this is the result we expect.

Usage:
    python tools/batch_variance_mcai.py <config_label>

Where <config_label> is e.g. "baseline_sglang_triton" or "dflash_sglang_triton".
Results merged into /tmp/batch_variance_mcai_results.json.
"""

from __future__ import annotations

import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests

URL = "http://127.0.0.1:30000"
MAX_NEW_TOKENS = 128
TIMEOUT_S = 600

TARGETS = [
    # Replaced the "What is 2+2?" prompt (2026-04-20): the 128 max_new_tokens
    # cap meant we were comparing 48 chars of "Thinking Process:" boilerplate,
    # not actual arithmetic. P3 below forces the model past template into real
    # multi-step reasoning — narrower token margins, genuinely tests
    # math-under-batch-variance.
    "A bag contains 5 red, 3 blue, and 2 green marbles. If I draw 3 marbles without replacement, what's the probability all three are different colors? Show your work.",
    "Write a Python function that reverses a string.",
    "Explain RSA encryption in one paragraph.",
]
# Two disjoint batchmate sets — same cardinality (3), different content.
# Divergence observed when swapping from A to B proves batch-content matters,
# not batch-size.
BATCHMATES_A = [
    "Count from 1 to 10.",
    "What is the capital of France?",
    "Summarize: AI is changing the world.",
]
BATCHMATES_B = [
    "Describe the weather in Tokyo.",
    "Name three primary colors.",
    "Explain photosynthesis briefly.",
]
# Extras to reach B=8
BATCHMATES_EXTRA = [
    "What year did WWII end?",
    "List three planets.",
    "Spell the word 'encyclopedia'.",
    "Translate 'hello' to Spanish.",
]

RESULTS_FILE = Path("/tmp/batch_variance_mcai_results.json")


def send_chat(prompt: str) -> str:
    """Single greedy completion via SGLang's OpenAI-compatible endpoint."""
    r = requests.post(
        URL + "/v1/chat/completions",
        json={
            "model": "Qwen/Qwen3.5-27B",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": MAX_NEW_TOKENS,
            "temperature": 0.0,
            "top_p": 1.0,
            "top_k": 1,
        },
        timeout=TIMEOUT_S,
    )
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]


def batched_burst(target_prompt: str, batchmates: list[str], target_slot: int) -> str:
    """Submit target + batchmates concurrently; return the target's output.

    Uses a ThreadPoolExecutor to submit all prompts nearly simultaneously so
    SGLang's continuous batching scheduler groups them into the same forward
    pass. The target is inserted at `target_slot`; other slots get batchmates.
    """
    prompts = list(batchmates)
    prompts.insert(target_slot, target_prompt)

    with ThreadPoolExecutor(max_workers=len(prompts)) as pool:
        futures = [pool.submit(send_chat, p) for p in prompts]
        outputs = [f.result() for f in futures]

    return outputs[target_slot]


def compare(solo: str, other: str) -> dict:
    """Structured comparison of two completions."""
    return {
        "bit_exact": solo == other,
        "solo_first_48": solo[:48],
        "other_first_48": other[:48],
        # Find first divergence position for post-hoc inspection
        "first_divergence_char": next(
            (i for i, (a, b) in enumerate(zip(solo, other)) if a != b), None
        )
        if solo != other
        else None,
    }


def run_batch_variance(label: str) -> dict:
    """Exercise the 5 batch configs per target, vs the target's solo output."""
    out: dict = {label: {}}
    for target in TARGETS:
        key = target[:40]
        print(f"[{label}] target: {key!r}")

        solo = send_chat(target)
        solo_2 = send_chat(target)
        self_consistent = solo == solo_2

        b4_mA_s0 = batched_burst(target, BATCHMATES_A, 0)
        b4_mB_s0 = batched_burst(target, BATCHMATES_B, 0)
        b4_mA_s3 = batched_burst(target, BATCHMATES_A, 3)
        b8_s0 = batched_burst(
            target, BATCHMATES_A + BATCHMATES_EXTRA, 0
        )  # B=8 total

        out[label][key] = {
            "solo_self_consistent": self_consistent,
            "B=4_batchmatesA_slot0": compare(solo, b4_mA_s0),
            "B=4_batchmatesB_slot0": compare(solo, b4_mB_s0),
            "B=4_batchmatesA_slot3": compare(solo, b4_mA_s3),
            "B=8_slot0": compare(solo, b8_s0),
        }
        print(
            f"  self-consistent: {self_consistent}  "
            f"B=4/A/s0: {out[label][key]['B=4_batchmatesA_slot0']['bit_exact']}  "
            f"B=4/B/s0: {out[label][key]['B=4_batchmatesB_slot0']['bit_exact']}  "
            f"B=4/A/s3: {out[label][key]['B=4_batchmatesA_slot3']['bit_exact']}  "
            f"B=8/s0: {out[label][key]['B=8_slot0']['bit_exact']}"
        )
    return out


def main() -> None:
    if len(sys.argv) != 2:
        print("Usage: python batch_variance_mcai.py <config_label>", file=sys.stderr)
        sys.exit(1)
    label = sys.argv[1]
    new = run_batch_variance(label)
    if RESULTS_FILE.exists():
        existing = json.loads(RESULTS_FILE.read_text())
        existing.update(new)
        new = existing
    RESULTS_FILE.write_text(json.dumps(new, indent=2))
    print(f"\nsaved to {RESULTS_FILE}")


if __name__ == "__main__":
    main()
