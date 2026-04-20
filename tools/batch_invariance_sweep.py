"""Expanded batch-variance sweep for the determinism blog (30 targets × 8 configs).

This is the "rate characterization" version of the pinpoint batch_variance_mcai.py
test. Same methodology (compare solo vs. batched greedy output at temp=0 for
each target prompt), but sampled across a spectrum of prompt types and batchmate
variations so the aggregate bit-exact rate is a meaningful population statistic
rather than anecdotal.

Categories (6 prompts each, 30 total):
    pure_math, code, factual_prose, template_cot, creative

Configs per target (8 batched comparisons):
    solo_self_consistent  — run solo twice, confirm B=1 stability (control)
    B=4 set A slot 0      — short factual batchmates
    B=4 set B slot 0      — medium instruction batchmates
    B=4 set C slot 0      — long prose batchmates
    B=4 set D slot 0      — math-adjacent batchmates (cross-domain test)
    B=4 set A slot 3      — position control (target in last slot)
    B=8 set E slot 0      — higher-concurrency stress

Total: 30 targets × (2 solo + 6 batched configs) = 30 × 7 SGLang requests minimum
(each batched config = one burst of 4-8 concurrent requests, server-side bundled).
Budget ~450-600 effective inference calls per server config.

Usage:
    python tools/batch_variance_sweep.py <config_label>           # live run
    python tools/batch_variance_sweep.py <config_label> --dry-run  # just print the matrix

Results are merged into /tmp/batch_variance_sweep_results.json.
"""

from __future__ import annotations

import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests

URL = os.environ.get("SERVER_URL", "http://127.0.0.1:30000")
MODEL_ID = os.environ.get("MODEL_ID", "Qwen/Qwen3.5-27B")
MAX_NEW_TOKENS = 128
TIMEOUT_S = 600
RESULTS_FILE = Path("/tmp/batch_variance_sweep_results.json")


# ============================================================================
# TARGETS — 30 prompts across 5 categories, 6 each.
# Keep within-category lengths roughly matched to avoid padding confounds.
# ============================================================================

TARGETS_BY_CATEGORY: dict[str, list[str]] = {
    "pure_math": [
        # P3 — already proven stable on MLX (retains complex-math canary)
        "A bag contains 5 red, 3 blue, and 2 green marbles. If I draw 3 marbles without replacement, what's the probability all three are different colors? Show your work.",
        "Solve for x: 3x\u00b2 + 7x - 20 = 0. Show your work.",
        "Compute: 127 \u00d7 43, then add 19, then divide by 8. Show your work.",
        "Is 1009 prime? Show your reasoning step by step.",
        "A train leaves city A at 60 mph, another at 80 mph from city B, 350 miles apart. When do they meet?",
        "Integrate: \u222b (3x\u00b2 + 2x - 5) dx, then evaluate from x=0 to x=4. Show your work.",
    ],
    "code": [
        "Write a Python function that reverses a string.",
        "Write a Python function that returns the nth Fibonacci number (0-indexed).",
        "Write a Rust function that checks if a string is a palindrome.",
        "Refactor this SQL to use CTEs: SELECT * FROM orders o JOIN customers c ON o.cust_id=c.id WHERE c.country='US';",
        "Implement binary search in Go, returning the index or -1.",
        "Debug: why does `for i in range(len(arr)): arr.pop(i)` fail? Explain and fix.",
    ],
    "factual_prose": [
        "Explain RSA encryption in one paragraph.",
        "Explain how the TCP three-way handshake works.",
        "Explain what a B-tree is and when you'd use one.",
        "Explain the difference between DNA and RNA in one paragraph.",
        "Describe photosynthesis in one paragraph.",
        "Describe the process of plate tectonics in one paragraph.",
    ],
    "template_cot": [
        "Think step by step, then solve: what year will be the 100th anniversary of 1969?",
        "Analyze this claim: 'AI will replace most programmers by 2030'. Walk through your reasoning.",
        "Given the statement 'All birds can fly', identify the logical flaw and explain.",
        "Evaluate this argument: 'If we allow X, then Y, and Y is bad, so X is bad.'",
        "Think carefully: is 0.999... equal to 1? Justify your answer.",
        "Walk through the logic: why is `True + True == 2` in Python?",
    ],
    "creative": [
        "Write a haiku about autumn rain.",
        "Write a one-paragraph opening for a noir detective story set in 2049.",
        "Write a brief dialogue between a skeptic and a believer about consciousness.",
        "Describe a city where gravity works sideways, in 50 words.",
        "Write a limerick about a database index.",
        "Draft a cover letter paragraph for a systems engineering role.",
    ],
}


# ============================================================================
# BATCHMATE SETS — A/B/C/D cardinality 3 (for B=4); E cardinality 7 (for B=8).
# Varied in length + domain so we probe different neighbor-shape scenarios.
# ============================================================================

BATCHMATES_A = [  # short factual
    "Count from 1 to 10.",
    "What is the capital of France?",
    "Name three prime numbers under 20.",
]
BATCHMATES_B = [  # medium instructions
    "Translate 'good morning' to Japanese.",
    "List three benefits of a good night's sleep.",
    "Summarize the plot of Hamlet in one line.",
]
BATCHMATES_C = [  # long prose (roughly ~200 char each)
    "Write two sentences describing a rainy afternoon in a coastal town, including one sensory detail and one emotional beat.",
    "In two sentences, explain why version control is useful for solo developers working across multiple machines over long time horizons.",
    "Describe, in two sentences, the experience of a cat watching birds through a window without being able to reach them.",
]
BATCHMATES_D = [  # math-adjacent (tests cross-domain batching)
    "Add: 234 + 567.",
    "What is 15% of 240?",
    "Convert 1 kilometer to miles, approximately.",
]
BATCHMATES_E = (BATCHMATES_A + BATCHMATES_B + BATCHMATES_D)[:7]  # 7 for B=8


# ============================================================================
# Inference helpers
# ============================================================================


def send_chat(prompt: str) -> str:
    """Single greedy completion via SGLang's OpenAI endpoint."""
    r = requests.post(
        URL + "/v1/chat/completions",
        json={
            "model": MODEL_ID,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": MAX_NEW_TOKENS,
            "temperature": 0.0,
            "top_p": 1.0,
            "top_k": 1,
        },
        timeout=TIMEOUT_S,
    )
    r.raise_for_status()
    msg = r.json()["choices"][0]["message"]
    return (msg.get("reasoning") or "") + (msg.get("content") or "")


def batched_burst(target_prompt: str, batchmates: list[str], target_slot: int) -> str:
    """Submit target + batchmates concurrently so SGLang batches them.

    Uses ThreadPoolExecutor to submit at nearly the same instant; SGLang's
    continuous-batching scheduler groups simultaneous arrivals into the same
    forward pass.
    """
    prompts = list(batchmates)
    prompts.insert(target_slot, target_prompt)
    with ThreadPoolExecutor(max_workers=len(prompts)) as pool:
        futures = [pool.submit(send_chat, p) for p in prompts]
        outputs = [f.result() for f in futures]
    return outputs[target_slot]


def compare(solo: str, other: str) -> dict:
    """Structured diff of two completions."""
    first_div = None
    if solo != other:
        first_div = next(
            (i for i, (a, b) in enumerate(zip(solo, other)) if a != b),
            min(len(solo), len(other)),
        )
    return {
        "bit_exact": solo == other,
        "solo_first_48": solo[:48],
        "other_first_48": other[:48],
        "first_divergence_char": first_div,
    }


# ============================================================================
# Sweep
# ============================================================================


def run_one_target(category: str, target: str) -> dict:
    """Run all 7 configs (1 self-consistency + 6 batched) for a single target."""
    solo = send_chat(target)
    solo_2 = send_chat(target)
    self_consistent = solo == solo_2

    result = {
        "category": category,
        "target": target,
        "solo_first_48": solo[:48],
        "solo_self_consistent": self_consistent,
        "B=4_setA_slot0": compare(solo, batched_burst(target, BATCHMATES_A, 0)),
        "B=4_setB_slot0": compare(solo, batched_burst(target, BATCHMATES_B, 0)),
        "B=4_setC_slot0": compare(solo, batched_burst(target, BATCHMATES_C, 0)),
        "B=4_setD_slot0": compare(solo, batched_burst(target, BATCHMATES_D, 0)),
        "B=4_setA_slot3": compare(solo, batched_burst(target, BATCHMATES_A, 3)),
        "B=8_setE_slot0": compare(solo, batched_burst(target, BATCHMATES_E, 0)),
    }
    return result


def aggregate(label_result: dict) -> dict:
    """Compute per-category and overall bit-exact rates."""
    by_category: dict[str, list[bool]] = {}
    all_flags: list[bool] = []

    batched_keys = [
        "B=4_setA_slot0",
        "B=4_setB_slot0",
        "B=4_setC_slot0",
        "B=4_setD_slot0",
        "B=4_setA_slot3",
        "B=8_setE_slot0",
    ]
    for target_data in label_result["targets"]:
        cat = target_data["category"]
        by_category.setdefault(cat, [])
        for k in batched_keys:
            flag = target_data[k]["bit_exact"]
            by_category[cat].append(flag)
            all_flags.append(flag)

    return {
        "by_category": {
            cat: {
                "bit_exact": sum(flags),
                "total": len(flags),
                "rate": sum(flags) / len(flags) if flags else 0.0,
            }
            for cat, flags in by_category.items()
        },
        "overall": {
            "bit_exact": sum(all_flags),
            "total": len(all_flags),
            "rate": sum(all_flags) / len(all_flags) if all_flags else 0.0,
        },
    }


def run_sweep(label: str) -> dict:
    """Run the full 30-target × 8-config sweep under the given server."""
    out: dict = {"label": label, "targets": []}
    start = time.time()
    idx = 0
    total = sum(len(ts) for ts in TARGETS_BY_CATEGORY.values())
    for category, targets in TARGETS_BY_CATEGORY.items():
        for target in targets:
            idx += 1
            print(f"[{label}] [{idx}/{total}] [{category}] {target[:48]!r}")
            result = run_one_target(category, target)
            out["targets"].append(result)
            flags = [
                result[k]["bit_exact"]
                for k in (
                    "B=4_setA_slot0",
                    "B=4_setB_slot0",
                    "B=4_setC_slot0",
                    "B=4_setD_slot0",
                    "B=4_setA_slot3",
                    "B=8_setE_slot0",
                )
            ]
            print(
                f"    solo_self={result['solo_self_consistent']}  batched bit-exact: {sum(flags)}/{len(flags)}"
            )
    out["elapsed_s"] = time.time() - start
    out["summary"] = aggregate(out)
    return out


def dry_run() -> None:
    """Print the matrix without hitting the network."""
    total_targets = sum(len(ts) for ts in TARGETS_BY_CATEGORY.values())
    batched_per_target = 6
    solo_per_target = 2
    print(f"sweep matrix: {total_targets} targets × (2 solo + 6 batched) configs")
    print(
        f"effective inference calls (per server config): {total_targets * (solo_per_target + batched_per_target)} individual + {total_targets * batched_per_target} bursts"
    )
    print()
    for category, targets in TARGETS_BY_CATEGORY.items():
        print(f"[{category}] ({len(targets)} targets)")
        for i, t in enumerate(targets, 1):
            print(f"  {i}. {t[:80]}{'...' if len(t) > 80 else ''}")
        print()
    print(
        f"batchmate sets: A={len(BATCHMATES_A)}, B={len(BATCHMATES_B)}, C={len(BATCHMATES_C)}, D={len(BATCHMATES_D)}, E={len(BATCHMATES_E)}"
    )


def main() -> None:
    if len(sys.argv) < 2:
        print(
            "Usage: python batch_variance_sweep.py <config_label> [--dry-run]",
            file=sys.stderr,
        )
        sys.exit(1)
    label = sys.argv[1]
    if "--dry-run" in sys.argv:
        dry_run()
        return

    new = run_sweep(label)
    # Top-level shape keyed by label for multi-server accumulation
    wrapped: dict = {label: new}
    if RESULTS_FILE.exists():
        existing = json.loads(RESULTS_FILE.read_text())
        existing.update(wrapped)
        wrapped = existing
    RESULTS_FILE.write_text(json.dumps(wrapped, indent=2))
    print(
        f"\nsummary ({label}): overall bit-exact = {new['summary']['overall']['bit_exact']}/{new['summary']['overall']['total']} ({new['summary']['overall']['rate']:.1%})"
    )
    for cat, stats in new["summary"]["by_category"].items():
        print(f"  {cat}: {stats['bit_exact']}/{stats['total']} ({stats['rate']:.1%})")
    print(f"\nsaved to {RESULTS_FILE}")


if __name__ == "__main__":
    main()
