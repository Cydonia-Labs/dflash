"""Verify that a running OpenAI-compatible LLM server batches concurrent requests.

Sends one solo request, measures T_solo; sends N concurrent requests as a
burst, measures T_batch. Ratio T_batch / T_solo tells us what happened:

  ~1.0-1.7x  → server batched into one forward pass (good)
  ~1.7-3.5x  → partial batching (investigate)
  >=3.5x     → server serialized the requests (batch-variance test invalid)

Run against a live server at SERVER_URL (default http://127.0.0.1:30000).
Used as a pre-flight check for the _mcai-suffixed HTTP-client experiments.
"""

from __future__ import annotations

import os
import time
from concurrent.futures import ThreadPoolExecutor

import requests

URL = os.environ.get("SERVER_URL", "http://127.0.0.1:30000")
MODEL_ID = os.environ.get("MODEL_ID", "Qwen/Qwen3.5-27B")
BURST_SIZE = 4
MAX_TOKENS = 64
PROMPT = "Write one short sentence about rain."
TIMEOUT_S = 600


def ping() -> float:
    t = time.time()
    r = requests.post(
        URL + "/v1/chat/completions",
        json={
            "model": MODEL_ID,
            "messages": [{"role": "user", "content": PROMPT}],
            "max_tokens": MAX_TOKENS,
            "temperature": 0.0,
            "top_p": 1.0,
            "top_k": 1,
        },
        timeout=TIMEOUT_S,
    )
    r.raise_for_status()
    msg = r.json()["choices"][0]["message"]
    (msg.get("reasoning") or "") + (msg.get("content") or "")
    return time.time() - t


def main() -> None:
    print(f"URL={URL}  model={MODEL_ID}  burst={BURST_SIZE}  max_tokens={MAX_TOKENS}")

    print("warm-up...")
    ping()

    print("solo run x2 (take min)...")
    solo = min(ping() for _ in range(2))

    print(f"concurrent burst of {BURST_SIZE}...")
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=BURST_SIZE) as pool:
        futures = [pool.submit(ping) for _ in range(BURST_SIZE)]
        for f in futures:
            f.result()
    batch = time.time() - t0

    ratio = batch / solo
    print(f"solo: {solo:.2f}s  burst-of-{BURST_SIZE}: {batch:.2f}s  ratio: {ratio:.2f}x")
    if ratio < 1.7:
        print("[ok] server is batching -- HTTP-client experiments are valid")
    elif ratio >= 3.5:
        print(
            "[fail] server appears to serialize -- HTTP-client batch-variance test "
            "is invalid; fall back to direct BatchGenerator"
        )
    else:
        print(
            "[warn] partial batching -- results may be harder to interpret; "
            "investigate --decode-concurrency / --prompt-concurrency"
        )


if __name__ == "__main__":
    main()
