"""Test B re-run with P3 (marble probability) substituted for the old 2+2 prompt.

Rationale: the original 2+2 result (~100% bit-exact under batching) was an
artifact — the model spent all 128 tokens in a 'Thinking Process:\n\n1.
**Analyze the Request:**' boilerplate template that happens to be stable under
batch-size perturbations. We never actually exercised arithmetic reasoning.

P3 forces real multi-step reasoning (inclusion-exclusion / combinatorics).
Prediction (pre-registered): P3 will show divergence comparable to
Python/RSA (roughly 30-60% bit-exact under batching), not the old ~100%.
If P3 matches the old math result, the template explanation is wrong.

Additions vs the original Test B:
  - solo output text (first ~200 chars) saved per target, so we can verify
    whether we got past template into actual reasoning.
  - for divergent cases, the decoded context window around first-divergence
    is saved (both sides), so we see the character where batch-variance
    flipped the argmax.
"""
import json
from pathlib import Path

from mlx_lm import load


MODEL = "mlx-community/Qwen3.5-27B-4bit"
MAX_NEW = 128

P3 = (
    "A bag contains 5 red, 3 blue, and 2 green marbles. If I draw 3 marbles "
    "without replacement, what's the probability all three are different colors? "
    "Show your work."
)

TARGETS = [
    P3,
    "Write a Python function that reverses a string.",
    "Explain RSA encryption in one paragraph.",
]

BATCHMATES_A = [
    "List the first five prime numbers and explain briefly.",
    "Summarize the plot of Hamlet in two sentences.",
    "Describe the water cycle for a fifth grader.",
]

BATCHMATES_B = [
    "Name three programming languages and their typical use.",
    "Define the term 'machine learning' in plain English.",
    "What are the primary colors?",
]

BATCHMATES_EXTRA = [
    "Give one example of a renewable energy source.",
    "What is photosynthesis?",
    "Explain gravity in one sentence.",
    "List three classic novels.",
]


def tokens_from_batch(model, tok, prompts, max_new):
    from mlx_lm.generate import BatchGenerator

    gen = BatchGenerator(
        model,
        stop_tokens=[[t] for t in tok.eos_token_ids],
    )
    max_tokens = [max_new] * len(prompts)
    uids = gen.insert([tok.encode(p, add_special_tokens=True) for p in prompts], max_tokens)
    results = {uid: [] for uid in uids}
    with gen.stats():
        while responses := gen.next_generated():
            for r in responses:
                if r.finish_reason != "stop":
                    results[r.uid].append(int(r.token))
    gen.close()
    return [results[uid] for uid in uids]


def compare(seq_a, seq_b, tok, window=20):
    first_div = None
    for i, (a, b) in enumerate(zip(seq_a, seq_b)):
        if a != b:
            first_div = i
            break
    out = {
        "len_a": len(seq_a),
        "len_b": len(seq_b),
        "bit_exact": seq_a == seq_b,
        "first_divergence": first_div,
        "common_prefix": first_div if first_div is not None else min(len(seq_a), len(seq_b)),
    }
    if first_div is not None:
        lo = max(0, first_div - window)
        hi_a = min(len(seq_a), first_div + window)
        hi_b = min(len(seq_b), first_div + window)
        out["prefix_text"] = tok.decode(seq_a[lo:first_div])
        out["solo_tail"] = tok.decode(seq_a[first_div:hi_a])
        out["batched_tail"] = tok.decode(seq_b[first_div:hi_b])
    return out


def main():
    print(f"Loading {MODEL} ...")
    model, tok = load(MODEL)

    results = {}
    for ti, target in enumerate(TARGETS):
        print(f"\n=== Target {ti}: {target[:60]!r}")
        per_target = {}

        print("  Solo run 1...")
        solo1 = tokens_from_batch(model, tok, [target], MAX_NEW)[0]
        solo1_text = tok.decode(solo1)
        print(f"    len={len(solo1)}")
        print(f"    first 200 chars: {solo1_text[:200]!r}")

        print("  Solo run 2...")
        solo2 = tokens_from_batch(model, tok, [target], MAX_NEW)[0]
        per_target["solo_self_consistency"] = compare(solo1, solo2, tok)
        print(f"    bit_exact={per_target['solo_self_consistency']['bit_exact']}")

        print("  Batched B=4 (batchmates A, target at slot 0)...")
        outputs_a = tokens_from_batch(model, tok, [target] + BATCHMATES_A, MAX_NEW)
        per_target["B4_batchmatesA_slot0"] = compare(solo1, outputs_a[0], tok)
        c = per_target["B4_batchmatesA_slot0"]
        print(f"    bit_exact={c['bit_exact']}  first_div={c['first_divergence']}")

        print("  Batched B=4 (batchmates B, target at slot 0)...")
        outputs_b = tokens_from_batch(model, tok, [target] + BATCHMATES_B, MAX_NEW)
        per_target["B4_batchmatesB_slot0"] = compare(solo1, outputs_b[0], tok)
        c = per_target["B4_batchmatesB_slot0"]
        print(f"    bit_exact={c['bit_exact']}  first_div={c['first_divergence']}")

        print("  Batched B=4 (batchmates A, target at slot 3)...")
        outputs_l = tokens_from_batch(model, tok, BATCHMATES_A + [target], MAX_NEW)
        per_target["B4_batchmatesA_slot3"] = compare(solo1, outputs_l[3], tok)
        c = per_target["B4_batchmatesA_slot3"]
        print(f"    bit_exact={c['bit_exact']}  first_div={c['first_divergence']}")

        print("  Batched B=8 (target at slot 0)...")
        all_others = BATCHMATES_A + BATCHMATES_B + BATCHMATES_EXTRA
        outputs_8 = tokens_from_batch(model, tok, [target] + all_others[:7], MAX_NEW)
        per_target["B8_target_slot0"] = compare(solo1, outputs_8[0], tok)
        c = per_target["B8_target_slot0"]
        print(f"    bit_exact={c['bit_exact']}  first_div={c['first_divergence']}")

        results[f"target_{ti}"] = {
            "prompt": target,
            "solo_tokens_len": len(solo1),
            "solo_text_first_200": solo1_text[:200],
            "comparisons": per_target,
        }

    out = Path("~/github/dflash/batch_invariance_b_p3_results.json").expanduser()
    out.write_text(json.dumps(results, indent=2))
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()
