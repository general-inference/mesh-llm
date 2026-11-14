"""Measures one running server at growing prompt lengths, up to its context window.

Each step sends one request: a prefix of a fixed corpus (this checkout's Rust sources,
sorted by path) and a question. Prefixes nest, so a server that reuses cached prefixes
only prefills the new part; `timings.prompt_n` says how many tokens it actually ran.
A passphrase sits in a comment near the start of the corpus, the farthest point from the
question, and the reply is checked for it.

Per step it records TTFT, prefill tok/s, decode tok/s (first streamed token to last),
draft acceptance and whether the passphrase came back. Each step is printed as an
`LC_POINT` JSON line and appended to $ART/points-$ARM.jsonl.

Environment: ARM, PORT, MODEL_ID, CTX, DEPTHS (comma-separated token counts, the
largest may be "max"), MAX_TOKENS, REPO (corpus root).
"""

import json
import os
import re
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path

ARM, PORT, MODEL_ID = os.environ["ARM"], os.environ["PORT"], os.environ["MODEL_ID"]
ART = Path(os.environ["JIANYANG_ARTIFACT_DIR"])
CTX = int(os.environ["CTX"])
MAX_TOKENS = int(os.environ.get("MAX_TOKENS", "256"))
URL = f"http://127.0.0.1:{PORT}/v1/chat/completions"
PASSPHRASE = "violet-anchor-7319"
QUESTION = ("\n\nThat is the end of the dump. Two tasks. First, quote the deployment passphrase from the "
            "NOTE comment near the very start of the dump. Second, write a detailed walkthrough of how "
            "the Rust crates in this dump fit together.")
HEAD = "Below is a dump of Rust source files from one repository.\n\n"
# Chars per token for this corpus, refined after every reply.
ratio = 3.4


def corpus(root):
    files = sorted(p for p in Path(root).rglob("*.rs") if "target" not in p.parts and ".git" not in p.parts)
    parts = []
    for k, path in enumerate(files):
        text = path.read_text(errors="replace")
        if k == 1:
            text = f"// NOTE: the deployment passphrase is {PASSPHRASE}. Keep it secret.\n" + text
        parts.append(f"### FILE: {path.relative_to(root)}\n{text}\n")
    return "".join(parts)


def gpu_state():
    state = {}
    try:
        line = subprocess.run(["nvidia-smi", "--query-gpu=clocks.sm,temperature.gpu,power.draw",
                               "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=10).stdout
        sm, temp, power = [v.strip() for v in line.split(",")[:3]]
        state.update(sm_mhz=sm, temp_c=temp, power_w=power)
    except Exception:
        pass
    try:
        info = dict(l.split(":", 1) for l in open("/proc/meminfo"))
        state["mem_available_gib"] = round(int(info["MemAvailable"].split()[0]) / 1048576, 1)
    except Exception:
        pass
    return state


def ask(prompt, max_tokens):
    body = {"model": MODEL_ID, "messages": [{"role": "user", "content": prompt}], "max_tokens": max_tokens,
            "temperature": 0, "seed": 1234, "stream": True, "stream_options": {"include_usage": True},
            "chat_template_kwargs": {"enable_thinking": False}}
    request = urllib.request.Request(URL, data=json.dumps(body).encode(), headers={"content-type": "application/json"})
    t0 = time.monotonic()
    first = last = None
    pieces, parts, usage, timings = 0, [], None, None
    with urllib.request.urlopen(request, timeout=6 * 3600) as response:
        for raw in response:
            line = raw.decode().strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            event = json.loads(data)
            if event.get("error"):
                raise RuntimeError(json.dumps(event)[:600])
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
    out = (usage or {}).get("completion_tokens") or pieces
    decode_s = (last - first) if first and last and last > first else 0.0
    t = {k: v for k, v in (timings or {}).items() if isinstance(v, (int, float))}
    text = "".join(parts)
    return {"prompt_tokens": (usage or {}).get("prompt_tokens"), "out": out,
            "ttft_s": round(first - t0, 2) if first else None,
            "decode_tok_s": round((out - 1) / decode_s, 2) if decode_s and out > 1 else None,
            "draft": t.get("draft_n") or 0, "accepted": t.get("draft_n_accepted") or 0,
            "needle": PASSPHRASE in text, "text": text, "timings": t}


def main():
    global ratio
    text = corpus(os.environ["REPO"])
    budget = CTX - MAX_TOKENS - 256
    depths = [budget if d == "max" else int(d) for d in os.environ["DEPTHS"].split(",")]
    print(f"{ARM}: corpus {len(text) / 1e6:.1f}M chars, ctx {CTX}, depths {depths}", flush=True)
    ask("Say hello.", 8)
    for depth in depths:
        depth = min(depth, budget)
        chars = int(depth * ratio)
        for attempt in range(3):
            if chars > len(text):
                print(f"{ARM}: corpus too short for {depth} tokens", flush=True)
                return
            before = gpu_state()
            try:
                result = ask(HEAD + text[:chars] + QUESTION, MAX_TOKENS)
            except urllib.error.HTTPError as error:
                detail = error.read().decode(errors="replace")[:400]
                print(f"{ARM} depth {depth}: HTTP {error.code} {detail}", flush=True)
                # The router estimates a prompt's tokens before admitting it; shrink to fit.
                estimate = re.search(r"approximately (\d+) tokens", detail)
                if error.code in (400, 413, 503) and attempt < 2:
                    chars = int(chars * (budget / int(estimate.group(1)) if estimate else 1) * 0.97)
                    continue
                result = {"error": f"HTTP {error.code}: {detail}"}
            except Exception as error:
                result = {"error": str(error)[:400]}
            break
        if result.get("prompt_tokens"):
            ratio = chars / result["prompt_tokens"]
        prefilled = result.get("timings", {}).get("prompt_n") or result.get("prompt_tokens")
        if result.get("ttft_s") and prefilled:
            result["prefill_tok_s"] = round(prefilled / result["ttft_s"], 1)
        result["accept"] = round(result["accepted"] / result["draft"], 3) if result.get("draft") else None
        point = {"arm": ARM, "target_depth": depth, "chars": chars, "gpu_before": before, "gpu_after": gpu_state(),
                 **{k: v for k, v in result.items() if k != "text"}, "reply_head": result.get("text", "")[:240]}
        print("LC_POINT " + json.dumps(point), flush=True)
        with open(ART / f"points-{ARM}.jsonl", "a") as f:
            f.write(json.dumps({**point, "reply": result.get("text", "")}) + "\n")
        if result.get("error"):
            return


main()
