"""Times uncached prefill of long prompts and plain decode against one mesh-llm server.

Each prompt opens with a unique request number, so no prompt shares a cached prefix
with an earlier one. Prefill time is the latency of a max_tokens=1 request.
"""
import json, os, time, urllib.error, urllib.request

PORT, MODEL, ARM = os.environ["PORT"], os.environ["MODEL_ID"], os.environ["ARM"]
URL = f"http://127.0.0.1:{PORT}/v1/chat/completions"
WORDS = ("the quick brown fox jumps over a lazy dog while seven wizards quietly "
         "judge boxing matches near the old harbour lighthouse at dawn").split()


def text(n_words, seed):
    return " ".join(WORDS[(i * 7 + seed) % len(WORDS)] for i in range(n_words))


def post(body, stream=False):
    req = urllib.request.Request(URL, json.dumps(body).encode(), {"content-type": "application/json"})
    try:
        return urllib.request.urlopen(req, timeout=900)
    except urllib.error.HTTPError as err:
        print(f"SERVE_BENCH_ERROR {ARM} HTTP {err.code}: {err.read().decode(errors='replace')[:2000]}", flush=True)
        raise


def prefill(n_words, seed):
    body = {"model": MODEL, "max_tokens": 1, "temperature": 0,
            "messages": [{"role": "user", "content": f"Request {seed}. " + text(n_words, seed) + "\nSummarise."}]}
    t = time.perf_counter()
    out = json.load(post(body))
    return time.perf_counter() - t, out.get("usage", {}).get("prompt_tokens", 0)


def decode(seed, n=256):
    body = {"model": MODEL, "max_tokens": n, "temperature": float(os.environ.get("SB_TEMP", "0")), "seed": seed, "stream": True,
            **({"top_k": int(os.environ["SB_TOPK"])} if os.environ.get("SB_TOPK") else {}),
            "messages": [{"role": "user", "content": f"Request {seed}. Write a Python module that implements a thread-safe LRU cache with tests."}]}
    stamps = []; texts = []
    with post(body) as r:
        for raw in r:
            line = raw.decode().strip()
            if line.startswith("data:") and "[DONE]" not in line:
                d = json.loads(line[5:])
                if d.get("choices") and (d["choices"][0].get("delta") or {}).get("content"):
                    stamps.append(time.perf_counter()); texts.append(d["choices"][0]["delta"]["content"])
    return (len(stamps) - 1) / (stamps[-1] - stamps[0]) if len(stamps) > 2 else 0.0, "".join(texts)


PHASES = os.environ.get("SB_PHASES", "warm long agent decode")
if "warm" in PHASES:
    prefill(200, 1)
seed = 100 + 50 * ("warm" not in PHASES)
for n_words in ((1500, 6000, 12000) if "decode" in PHASES else (12000,)) if "long" in PHASES else ():
    runs = []
    for _ in range(3):
        seed += 1
        runs.append(prefill(n_words, seed))
    best = min(r[0] for r in runs)
    toks = runs[0][1]
    print("SERVE_BENCH " + json.dumps({"arm": ARM, "kind": "prefill", "prompt_tokens": toks,
          "best_s": round(best, 3), "tok_s": round(toks / best, 1), "all_s": [round(r[0], 3) for r in runs]}), flush=True)
if "agent" in PHASES:
    # Agent turns: a long history, then the same history plus a short new message. The second
    # request hits the prefix cache, so its latency is the suffix prefill plus whatever the
    # server does around it (state export for the cache, restore).
    for n_words in (6000, 12000):
        seed += 1
        history = f"Session {seed}. " + text(n_words, seed)
        def turn(extra):
            # The history sits in earlier messages, as in an agent loop, so the new turn
            # starts at a message boundary the server can checkpoint.
            body = {"model": MODEL, "max_tokens": 1, "temperature": 0,
                    "messages": [{"role": "user", "content": history},
                                 {"role": "assistant", "content": "Noted."},
                                 {"role": "user", "content": extra}]}
            t = time.perf_counter()
            out = json.load(post(body))
            return time.perf_counter() - t, out.get("usage", {}).get("prompt_tokens", 0)
        first, toks = turn("\nWhat happened first?")
        hits = [turn(f"\nWhat happened next? ({i})")[0] for i in range(3)]
        print("SERVE_BENCH " + json.dumps({"arm": ARM, "kind": "agent_turn", "prompt_tokens": toks,
              "first_s": round(first, 3), "cached_turn_s": [round(h, 3) for h in hits]}), flush=True)
if "agentloop" in PHASES:
    # An append-style agent loop: each turn sends the whole conversation so far, including the
    # model's own previous reply, plus a new user message. TTFT per turn is what an agent waits
    # for between tool calls; turns after the first should start from the cached conversation.
    def stream_turn(messages, n):
        body = {"model": MODEL, "max_tokens": n, "temperature": 0, "stream": True, "messages": messages}
        t = time.perf_counter(); first = None; out = []
        with post(body) as r:
            for raw in r:
                line = raw.decode().strip()
                if line.startswith("data:") and "[DONE]" not in line:
                    d = json.loads(line[5:])
                    piece = (d.get("choices") or [{}])[0].get("delta", {}).get("content")
                    if piece:
                        first = first or time.perf_counter()
                        out.append(piece)
        return (first or time.perf_counter()) - t, "".join(out)
    for n_words in (6000, 12000):
        seed += 1
        messages = [{"role": "user", "content": f"Session {seed}. " + text(n_words, seed) + "\nRead this log."}]
        ttfts = []
        for turn_i in range(5):
            ttft, reply = stream_turn(messages, 48)
            ttfts.append(round(ttft, 3))
            messages = messages + [{"role": "assistant", "content": reply},
                                   {"role": "user", "content": f"Tool result {turn_i}: ok. What next?"}]
        print("SERVE_BENCH " + json.dumps({"arm": ARM, "kind": "agent_loop", "history_words": n_words,
              "turn_ttft_s": ttfts}), flush=True)
if "decode" not in PHASES:
    raise SystemExit(0)
import hashlib
runs = [decode(500 + i) for i in range(int(os.environ.get("SB_DECODE_RUNS", "3")))]
rates = [r for r, _ in runs]
print("SERVE_BENCH " + json.dumps({"arm": ARM, "kind": "decode_text", "sha": [hashlib.sha256(t.encode()).hexdigest()[:12] for _, t in runs]}), flush=True)
print("SERVE_BENCH " + json.dumps({"arm": ARM, "kind": "decode", "tok_s": round(max(rates), 2), "all": [round(r, 2) for r in rates]}), flush=True)
