"""Measures one running SGLang server over the spec-prompts or agent-transcript set.

Same inputs and decode-speed definition as dflash-defaults/measure.py (tokens after the
first chunk over time from first to last chunk, greedy, 256 tokens), but through SGLang's
native /generate so each reply carries spec_verify_ct and acceptance length is exact.
At concurrency > 1 a pool of N workers drains the list; aggregate throughput is total
output tokens over the pass's wall time.
"""

import argparse
import concurrent.futures as cf
import hashlib
import json
import statistics
import threading
import time
import urllib.request

ap = argparse.ArgumentParser()
ap.add_argument("--port", type=int, required=True)
ap.add_argument("--prompts", required=True)
ap.add_argument("--split", default="report")
ap.add_argument("--tokenizer", required=True)
ap.add_argument("--concurrency", type=int, default=1)
ap.add_argument("--max-tokens", type=int, default=256)
ap.add_argument("--label", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--limit", type=int, default=0)
args = ap.parse_args()

from transformers import AutoTokenizer  # noqa: E402

tok = AutoTokenizer.from_pretrained(args.tokenizer)
items = [json.loads(line) for line in open(args.prompts) if line.strip()]
items = [i for i in items if args.split == "all" or i["split"] == args.split]
if args.limit:
    items = items[: args.limit]
URL = f"http://127.0.0.1:{args.port}/generate"


def group(item):
    if item.get("group"):
        return item["group"]
    c = item["category"]
    return "code" if c in ("code", "coding") else "math" if c == "math" else "prose"


def once(prompt):
    text = tok.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True)
    body = {"text": text, "stream": True,
            "sampling_params": {"temperature": 0.0, "max_new_tokens": args.max_tokens}}
    req = urllib.request.Request(URL, data=json.dumps(body).encode(), headers={"content-type": "application/json"})
    t0 = time.monotonic()
    first = last = None
    first_tokens = 0
    meta = {}
    final_text = ""
    with urllib.request.urlopen(req, timeout=3600) as resp:
        for raw in resp:
            line = raw.decode().strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            ev = json.loads(data)
            if "error" in ev:
                raise RuntimeError(json.dumps(ev)[:500])
            m = ev.get("meta_info") or {}
            ct = m.get("completion_tokens") or 0
            if ct > (meta.get("completion_tokens") or 0):
                now = time.monotonic()
                if first is None:
                    first, first_tokens = now, ct
                last = now
            meta = m or meta
            final_text = ev.get("text", final_text)
    out = meta.get("completion_tokens") or 0
    dec_s = (last - first) if first and last and last > first else 0.0
    return {"out": out, "prompt": meta.get("prompt_tokens"), "ttft_ms": (first - t0) * 1000 if first else None,
            "dec_tokens": max(out - first_tokens, 0) if dec_s > 0 else 0, "dec_s": dec_s,
            "e2e_s": time.monotonic() - t0, "t_start": t0, "t_end": time.monotonic(),
            "verify_ct": meta.get("spec_verify_ct") or 0, "accept_length": meta.get("spec_accept_length"),
            "cached_tokens": meta.get("cached_tokens"),
            "sha": hashlib.sha256(final_text.encode()).hexdigest()[:12]}


once(items[0]["prompt"][:2000])  # warm-up, not measured
results = [None] * len(items)
lock = threading.Lock()
done = [0]


def work(k):
    r = once(items[k]["prompt"])
    r.update(id=items[k]["id"], group=group(items[k]))
    results[k] = r
    with lock:
        done[0] += 1
        if done[0] % 10 == 0:
            print(f"{args.label} {done[0]}/{len(items)}", flush=True)


wall0 = time.monotonic()
with cf.ThreadPoolExecutor(args.concurrency) as pool:
    list(pool.map(work, range(len(items))))
wall = time.monotonic() - wall0

groups = {}
names = list(dict.fromkeys(r["group"] for r in results))
order = ["prose", "code", "math", "4-8K", "8-16K", "16-32K", "32-48K"]
for name in sorted(names, key=lambda n: order.index(n) if n in order else 99) + ["all"]:
    rows = [r for r in results if name == "all" or r["group"] == name]
    dec_s = sum(r["dec_s"] for r in rows)
    toks = sum(r["dec_tokens"] for r in rows)
    vct = sum(r["verify_ct"] for r in rows)
    ttfts = sorted(r["ttft_ms"] for r in rows if r["ttft_ms"] is not None)
    per_req = [r["dec_tokens"] / r["dec_s"] for r in rows if r["dec_s"] > 0]
    groups[name] = {"n": len(rows), "tok_s": toks / dec_s if dec_s else None,
                    "per_request_tok_s_mean": statistics.mean(per_req) if per_req else None,
                    "accept_length": sum(r["out"] for r in rows) / vct if vct else None,
                    "ttft_ms_p50": ttfts[len(ttfts) // 2] if ttfts else None,
                    "out_tokens": sum(r["out"] for r in rows),
                    "prompt_tokens_mean": sum(r["prompt"] or 0 for r in rows) / len(rows)}
groups["all"]["wall_s"] = wall
groups["all"]["agg_out_tok_s"] = groups["all"]["out_tokens"] / wall
summary = {"label": args.label, "concurrency": args.concurrency, "split": args.split, "groups": groups}
with open(args.out, "a") as f:
    f.write(json.dumps({**summary, "results": results}) + "\n")
g = groups["all"]
print(f"DSPINE_AB {json.dumps(summary)}", flush=True)
print(f"{args.label} c={args.concurrency}: decode {g['tok_s'] or 0:.1f} tok/s/stream, agg {g['agg_out_tok_s']:.1f} tok/s, "
      f"AL {g['accept_length'] or 0:.2f}, TTFT p50 {g['ttft_ms_p50'] or 0:.0f} ms", flush=True)
