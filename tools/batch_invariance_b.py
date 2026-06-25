"""Test B: generation batch-variance on MLX.

Does running prompt P alone produce the same *generated tokens* as running P
as part of a batch with other prompts? Temp=0, argmax sampler, 128 new tokens.

Uses mlx_lm.batch_generate (continuous batching, per-sample KV caches).

For each target prompt we produce:
  - solo:    batch_generate([target])          -> O_solo
  - batchA:  batch_generate([target, q1,q2,q3]) -> O_A (position 0)
  - batchB:  batch_generate([target, r1,r2,r3]) -> O_B (different batchmates)
  - batchL:  batch_generate([q1,q2,q3, target]) -> O_L (target at last position)
  - batch8:  batch_generate([target, q1..q7])  -> O_8 (B=8)

Compare all non-solo sequences against O_solo for:
  - bit-exact sequence equality
  - first-divergence step (if any)
  - final token count
"""
import json
from pathlib import Path

from mlx_lm import load, batch_generate


MODEL = "mlx-community/Qwen3.5-27B-4bit"
MAX_NEW = 128

TARGETS = [
    "What is 2+2? Think step by step, then answer.",
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


def tokens_from_batch(model, tok, prompts, max_new, return_caches=False):
    """Run batch_generate and return per-prompt output token lists.

    We re-implement by capturing the low-level stream so we get token IDs
    (batch_generate returns decoded text). Approach: call batch_generate with
    a small hack — pass through verbose=False, then decode text and re-encode.
    That is lossy. Instead, use BatchGenerator directly.
    """
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


def compare(seq_a, seq_b):
    # First divergence
    first_div = None
    for i, (a, b) in enumerate(zip(seq_a, seq_b)):
        if a != b:
            first_div = i
            break
    return {
        "len_a": len(seq_a),
        "len_b": len(seq_b),
        "bit_exact": seq_a == seq_b,
        "first_divergence": first_div,
        "common_prefix": first_div if first_div is not None else min(len(seq_a), len(seq_b)),
    }


def main():
    print(f"Loading {MODEL} ...")
    model, tok = load(MODEL)

    results = {}
    for ti, target in enumerate(TARGETS):
        print(f"\n=== Target {ti}: {target[:50]!r}")
        per_target = {}

        # Solo twice to confirm self-consistency
        print("  Solo run 1...")
        solo1 = tokens_from_batch(model, tok, [target], MAX_NEW)[0]
        print(f"    len={len(solo1)}")
        print("  Solo run 2...")
        solo2 = tokens_from_batch(model, tok, [target], MAX_NEW)[0]
        per_target["solo_self_consistency"] = compare(solo1, solo2)
        print(f"    bit_exact={per_target['solo_self_consistency']['bit_exact']}")

        # Batched: target + BATCHMATES_A (B=4, target at slot 0)
        print("  Batched B=4 (batchmates A, target at slot 0)...")
        outputs_a = tokens_from_batch(model, tok, [target] + BATCHMATES_A, MAX_NEW)
        per_target["B4_batchmatesA_slot0"] = compare(solo1, outputs_a[0])
        print(f"    bit_exact={per_target['B4_batchmatesA_slot0']['bit_exact']}  first_div={per_target['B4_batchmatesA_slot0']['first_divergence']}")

        # Batched: target + BATCHMATES_B (different batchmates, still B=4, slot 0)
        print("  Batched B=4 (batchmates B, target at slot 0)...")
        outputs_b = tokens_from_batch(model, tok, [target] + BATCHMATES_B, MAX_NEW)
        per_target["B4_batchmatesB_slot0"] = compare(solo1, outputs_b[0])
        print(f"    bit_exact={per_target['B4_batchmatesB_slot0']['bit_exact']}  first_div={per_target['B4_batchmatesB_slot0']['first_divergence']}")

        # Batched: BATCHMATES_A + target (B=4, target at last slot)
        print("  Batched B=4 (batchmates A, target at slot 3)...")
        outputs_l = tokens_from_batch(model, tok, BATCHMATES_A + [target], MAX_NEW)
        per_target["B4_batchmatesA_slot3"] = compare(solo1, outputs_l[3])
        print(f"    bit_exact={per_target['B4_batchmatesA_slot3']['bit_exact']}  first_div={per_target['B4_batchmatesA_slot3']['first_divergence']}")

        # Batched B=8
        print("  Batched B=8 (target at slot 0)...")
        all_others = BATCHMATES_A + BATCHMATES_B + BATCHMATES_EXTRA
        outputs_8 = tokens_from_batch(model, tok, [target] + all_others[:7], MAX_NEW)
        per_target["B8_target_slot0"] = compare(solo1, outputs_8[0])
        print(f"    bit_exact={per_target['B8_target_slot0']['bit_exact']}  first_div={per_target['B8_target_slot0']['first_divergence']}")

        results[f"target_{ti}"] = {
            "prompt": target,
            "solo_tokens_len": len(solo1),
            "comparisons": per_target,
        }

    out = Path("~/github/dflash/batch_invariance_b_results.json").expanduser()
    out.write_text(json.dumps(results, indent=2))
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()
