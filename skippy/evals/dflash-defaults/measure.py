"""Measures single-stream decode speed of one running server over the spec-prompts set.

Decode speed is total decoded tokens over total decode time, from each reply's first
streamed token to its last, so prompt processing and model load are excluded. One
warm-up request runs first. The per-arm summary is printed as a `DFLASH_BENCH` JSON
line, which `bench.py table` reads back out of the job log.
"""

import hashlib
import json
import os
import time
import urllib.error
import urllib.request

ARM, PORT, MODEL_ID = os.environ["ARM"], os.environ["PORT"], os.environ["MODEL_ID"]
ART = os.environ["JIANYANG_ARTIFACT_DIR"]
TEMP = float(os.environ.get("TEMP") or 0)
SPLIT = os.environ.get("SPLIT", "report")
MAX_TOKENS = int(os.environ.get("MAX_TOKENS", "256"))
URL = f"http://127.0.0.1:{PORT}/v1/chat/completions"

items = [json.loads(line) for line in open(os.environ["PROMPTS"]) if line.strip()]
items = [item for item in items if SPLIT == "all" or item["split"] == SPLIT]


def group(item):
    """A prompt set may name its own groups (the agent set uses context bands)."""
    if item.get("group"):
        return item["group"]
    category = item["category"]
    return "code" if category in ("code", "coding") else "math" if category == "math" else "prose"


def once(prompt, seed):
    body = {"model": MODEL_ID, "messages": [{"role": "user", "content": prompt}], "max_tokens": MAX_TOKENS,
            "temperature": TEMP, "seed": seed, "stream": True, "stream_options": {"include_usage": True}}
    if TEMP > 0:
        body["top_p"] = 0.95
    request = urllib.request.Request(URL, data=json.dumps(body).encode(), headers={"content-type": "application/json"})
    t0 = time.monotonic()
    first = last = None
    pieces = 0
    usage = timings = None
    parts = []
    data = ""
    with urllib.request.urlopen(request, timeout=900) as response:
        for raw in response:
            line = raw.decode().strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            event = json.loads(data)
            if event.get("error") and not EMPTY_REPLIES:
                EMPTY_REPLIES.append(json.dumps(event)[:600])
            usage = event.get("usage") or usage
            timings = event.get("timings") or timings
            for choice in event.get("choices") or []:
                delta = choice.get("delta") or {}
                piece = (delta.get("reasoning_content") or "") + (delta.get("content") or "")
                if piece:
                    now = time.monotonic()
                    first = first or now
                    last = now
                    pieces += 1
                    parts.append(piece)
    if not pieces and not EMPTY_REPLIES:
        EMPTY_REPLIES.append(f"no content; last data: {data[:600]}")
    out = (usage or {}).get("completion_tokens") or pieces
    decode_s = (last - first) if first and last and last > first else 0.0
    t = timings or {}
    return {"out": out, "prompt": (usage or {}).get("prompt_tokens"), "ttft_ms": (first - t0) * 1000 if first else None,
            # The first token arrives at `first`, so the timed span decodes out - 1 tokens.
            "dec_tokens": max(out - 1, 0) if decode_s > 0 else 0, "dec_s": decode_s, "e2e_s": time.monotonic() - t0,
            "draft": t.get("draft_n") or 0, "accepted": t.get("draft_n_accepted") or 0,
            "sha": hashlib.sha256("".join(parts).encode()).hexdigest()[:12],
            "timings": {k: v for k, v in t.items() if isinstance(v, (int, float))}}


# The first reply that carried an error or no content, so a failed arm says why.
EMPTY_REPLIES = []
# A quick profile can measure the first LIMIT prompts, taken round-robin across groups.
if int(os.environ.get("LIMIT") or 0):
    by_group = {}
    for item in items:
        by_group.setdefault(group(item), []).append(item)
    picked = []
    while len(picked) < int(os.environ["LIMIT"]) and any(by_group.values()):
        for rows in by_group.values():
            if rows and len(picked) < int(os.environ["LIMIT"]):
                picked.append(rows.pop(0))
    items = picked
# A split can report every stage ready a moment before requests route to it; until then
# the API answers 503. Retry the warm-up through that window instead of failing the arm.
for attempt in range(60):
    try:
        once(items[0]["prompt"], 0)
        break
    except urllib.error.HTTPError as error:
        if error.code != 503 or attempt == 59:
            raise
        time.sleep(5)
results = []
for k, item in enumerate(items):
    result = once(item["prompt"], 1234 + k)
    result.update(id=item["id"], category=item.get("category", ""), group=group(item))
    results.append(result)
    if (k + 1) % 20 == 0:
        print(f"{ARM} {k + 1}/{len(items)}", flush=True)

groups = {}
names = list(dict.fromkeys(r["group"] for r in results))
for name in sorted(names, key=lambda n: ["prose", "code", "math"].index(n) if n in ("prose", "code", "math") else 9) + ["all"]:
    rows = [r for r in results if name == "all" or r["group"] == name]
    if not rows:
        continue
    decode_s = sum(r["dec_s"] for r in rows)
    tokens = sum(r["dec_tokens"] for r in rows)
    drafted = sum(r["draft"] for r in rows)
    ttfts = sorted(r["ttft_ms"] for r in rows if r["ttft_ms"] is not None)
    groups[name] = {"n": len(rows), "tok_s": tokens / decode_s if decode_s else None, "dec_tokens": tokens,
                    "dec_s": decode_s, "accept": sum(r["accepted"] for r in rows) / drafted if drafted else None,
                    "ttft_ms_p50": ttfts[len(ttfts) // 2] if ttfts else None,
                    "prompt_tokens_mean": sum(r["prompt"] or 0 for r in rows) / len(rows)}
    print(f"{ARM:18s} T={TEMP} {SPLIT:6s} {name:5s} n={len(rows):3d} decode {groups[name]['tok_s'] or 0:6.1f} tok/s"
          f"  accept {groups[name]['accept'] or 0:.2f}", flush=True)

with open(os.path.join(ART, "arms.jsonl"), "a") as f:
    f.write(json.dumps({"arm": ARM, "temp": TEMP, "split": SPLIT, "groups": groups, "results": results}) + "\n")
# Totals of the server's speculation timings over the arm, to show where a round's time goes.
spec_keys = ("window", "verify", "proposal", "downstream", "draft", "decode", "speculative")
totals = {}
for r in results:
    for k, v in r["timings"].items():
        if any(word in k for word in spec_keys):
            totals[k] = totals.get(k, 0) + v
if totals:
    print("SPEC_TIMINGS " + json.dumps({"arm": ARM, "requests": len(results), "totals": totals}), flush=True)
if EMPTY_REPLIES:
    print(f"{ARM}: first empty reply: {EMPTY_REPLIES[0]}", flush=True)
print("DFLASH_BENCH " + json.dumps({"arm": ARM, "temp": TEMP, "split": SPLIT, "groups": groups}), flush=True)
