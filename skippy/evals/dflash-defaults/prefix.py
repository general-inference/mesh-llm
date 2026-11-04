#!/usr/bin/env python3
"""Measures prefix-cache reuse on a served model: a second agent turn against a cold one.

For each agent transcript that fits the context, three requests go to the server:
  cold   a unique marker, then the whole history: nothing to reuse
  prime  a second marker, then the first PRIME_FRACTION of the history (the earlier turn)
  warm   the second marker, then the whole history: the earlier turn is a shared prefix
Prints PREFIX_BENCH with time to first token for cold and warm, and the server's
prefix-cache timings (hit tokens, bytes sent and saved) where it reports them.

Environment: PORT, MODEL_ID, PROMPTS (agent-transcripts.jsonl), MAX_PROMPT_CHARS,
PRIME_FRACTION (default 0.7), LIMIT (transcripts, default 8), ART (output directory).
"""

import json
import os
import statistics
import time
import urllib.request

PORT = os.environ["PORT"]
MODEL_ID = os.environ["MODEL_ID"]
URL = f"http://127.0.0.1:{PORT}/v1/chat/completions"
MAX_PROMPT_CHARS = int(os.environ.get("MAX_PROMPT_CHARS", "24000"))
PRIME_FRACTION = float(os.environ.get("PRIME_FRACTION", "0.7"))
LIMIT = int(os.environ.get("LIMIT") or 8)


def first_token(prompt):
    body = {"model": MODEL_ID, "messages": [{"role": "user", "content": prompt}], "max_tokens": 16,
            "temperature": 0, "stream": True, "stream_options": {"include_usage": True}}
    request = urllib.request.Request(URL, data=json.dumps(body).encode(), headers={"content-type": "application/json"})
    t0 = time.monotonic()
    ttft = None
    timings = {}
    with urllib.request.urlopen(request, timeout=900) as response:
        for raw in response:
            line = raw.decode().strip()
            if not line.startswith("data:") or line[5:].strip() == "[DONE]":
                continue
            event = json.loads(line[5:])
            timings = event.get("timings") or timings
            if ttft is None and any((c.get("delta") or {}).get("content") or (c.get("delta") or {}).get("reasoning_content")
                                    for c in event.get("choices") or []):
                ttft = (time.monotonic() - t0) * 1000
    return ttft, {k: v for k, v in timings.items() if isinstance(v, (int, float)) and "prefix" in k}


items = [json.loads(line) for line in open(os.environ["PROMPTS"]) if line.strip()]
items = [item for item in items if len(item["prompt"]) <= MAX_PROMPT_CHARS][:LIMIT]
rows = []
for k, item in enumerate(items):
    history = item["prompt"]
    cold_ttft, cold_t = first_token(f"[session cold-{k}-{time.time_ns()}]\n{history}")
    marker = f"[session warm-{k}-{time.time_ns()}]\n"
    first_token(marker + history[: int(len(history) * PRIME_FRACTION)])
    warm_ttft, warm_t = first_token(marker + history)
    rows.append({"id": item["id"], "chars": len(history), "cold_ms": cold_ttft, "warm_ms": warm_ttft,
                 "cold": cold_t, "warm": warm_t})
    print(f"prefix {k + 1}/{len(items)} chars={len(history)} cold={cold_ttft:.0f}ms warm={warm_ttft:.0f}ms "
          f"hit={warm_t.get('prefix_cache_hit_tokens')}", flush=True)

summary = {"n": len(rows), "cold_ttft_ms_p50": statistics.median(r["cold_ms"] for r in rows),
           "warm_ttft_ms_p50": statistics.median(r["warm_ms"] for r in rows),
           "warm_hit_tokens_mean": statistics.mean(r["warm"].get("prefix_cache_hit_tokens", 0) for r in rows),
           "warm_bytes_saved_mean": statistics.mean(r["warm"].get("prefix_cache_bytes_saved", 0) for r in rows),
           "cold_prefill_bytes_mean": statistics.mean(r["cold"].get("prefix_cache_prefill_bytes_sent", 0) for r in rows),
           "warm_prefill_bytes_mean": statistics.mean(r["warm"].get("prefix_cache_prefill_bytes_sent", 0) for r in rows)}
with open(os.path.join(os.environ.get("ART", "."), "prefix.jsonl"), "a") as f:
    f.write(json.dumps({"rows": rows, "summary": summary}) + "\n")
print("PREFIX_BENCH " + json.dumps(summary), flush=True)
