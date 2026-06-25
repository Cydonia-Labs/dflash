"""Test A: logits batch-variance on MLX.

Question: does forwarding prompt P alone produce the same final-position logits
as forwarding P alongside other prompts in a batch?

Design constraints from the MLX Qwen3 model:
- model.__call__ does NOT accept an attention mask; causal-only.
- So left-padding confounds the test (pad tokens attend as real).
- Solution: truncate all prompts to an identical token length; no padding.

For batch size B we:
- Encode B prompts, truncate each to min token length among them.
- Run each solo: logits_solo[i] = model(mx.array([prompt_i]))[0, -1, :]
- Run batched: logits_batched[i] = model(mx.array(prompts))[i, -1, :]
- Compare solo vs batched for each i.

Metrics:
- bit-exact equality of the full logit vector
- argmax equality (i.e. does the sampled next token match)
- max absolute diff in logit magnitude
"""
import json
from pathlib import Path

import numpy as np
import mlx.core as mx
from mlx_lm import load


MODEL = "mlx-community/Qwen3.5-27B-4bit"

PROMPTS = [
    "What is 2+2? Think step by step, then answer.",
    "Write a Python function that reverses a string.",
    "Explain RSA encryption in one paragraph.",
    "List the first five prime numbers and explain briefly.",
    "Summarize the plot of Hamlet in two sentences.",
    "Describe the water cycle for a fifth grader.",
    "Name three programming languages and their typical use.",
    "Define the term 'machine learning' in plain English.",
]

BATCH_SIZES = [2, 4, 8]


def encode_all(tok, prompts, min_len):
    toks = [tok.encode(p, add_special_tokens=True)[:min_len] for p in prompts]
    return toks


def forward_last_logits(model, token_rows):
    """Return logits at the last position for each row, shape (B, V)."""
    x = mx.array(token_rows)
    out = model(x)
    last = out[:, -1, :]
    mx.eval(last)
    return np.array(last.astype(mx.float32))


def main():
    print(f"Loading {MODEL} ...")
    model, tok = load(MODEL)

    # Decide min token length across the prompt set
    all_tokens = [tok.encode(p, add_special_tokens=True) for p in PROMPTS]
    min_len = min(len(t) for t in all_tokens)
    trimmed = [t[:min_len] for t in all_tokens]
    print(f"All prompts truncated to {min_len} tokens.")

    # Solo logits once per prompt
    print("Computing solo logits...")
    solo_logits = {i: forward_last_logits(model, [trimmed[i]])[0] for i in range(len(trimmed))}

    results = {"min_len": min_len, "per_batch_size": {}}

    for B in BATCH_SIZES:
        if B > len(trimmed):
            continue
        print(f"\nBatch size B={B}")
        batch_rows = trimmed[:B]
        batched = forward_last_logits(model, batch_rows)
        per_row = []
        for i in range(B):
            s = solo_logits[i]
            b = batched[i]
            bit_equal = bool(np.array_equal(s, b))
            argmax_equal = bool(int(s.argmax()) == int(b.argmax()))
            max_abs_diff = float(np.max(np.abs(s - b)))
            top5_solo = [int(x) for x in np.argsort(s)[-5:][::-1]]
            top5_batched = [int(x) for x in np.argsort(b)[-5:][::-1]]
            per_row.append({
                "prompt_idx": i,
                "prompt_preview": PROMPTS[i][:40],
                "bit_equal": bit_equal,
                "argmax_equal": argmax_equal,
                "max_abs_diff": max_abs_diff,
                "top5_solo": top5_solo,
                "top5_batched": top5_batched,
            })
            print(f"  [{i}] {PROMPTS[i][:40]!r}")
            print(f"      bit_equal={bit_equal}  argmax_equal={argmax_equal}  max|diff|={max_abs_diff:.3e}")
        results["per_batch_size"][B] = per_row

    # Position-dependence: put prompt 0 at different slots in B=4
    print("\nPosition test (prompt 0 at slots 0,1,2,3 in B=4)")
    pos_results = []
    for slot in range(4):
        rows = [trimmed[j] for j in [1, 2, 3, 4]]  # 4 distinct non-target prompts
        rows[slot] = trimmed[0]  # place prompt 0 at `slot`
        logits = forward_last_logits(model, rows)
        b = logits[slot]
        s = solo_logits[0]
        pos_results.append({
            "slot": slot,
            "bit_equal": bool(np.array_equal(s, b)),
            "argmax_equal": bool(int(s.argmax()) == int(b.argmax())),
            "max_abs_diff": float(np.max(np.abs(s - b))),
        })
        print(f"  slot={slot}  bit_equal={pos_results[-1]['bit_equal']}  argmax_equal={pos_results[-1]['argmax_equal']}  max|diff|={pos_results[-1]['max_abs_diff']:.3e}")
    results["position_test_B4"] = pos_results

    out = Path("~/github/dflash/batch_invariance_a_results.json").expanduser()
    out.write_text(json.dumps(results, indent=2))
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()
