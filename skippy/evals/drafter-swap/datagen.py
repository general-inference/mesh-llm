"""Has the target answer the training prompts of some domains; writes SpecForge-format jsonl.

Run by datagen.sh against a running server. Greedy, thinking off, like the eval arms,
so the draft learns the text it is measured on. Requests run CONCURRENCY at a time.
One file per domain, `train-<domain>.jsonl`, each line
`{"id", "conversations": [{"role": "user", ...}, {"role": "assistant", ...}]}`.
"""

import json
import os
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

PORT, MODEL_ID = os.environ["PORT"], os.environ["MODEL_ID"]
ART = os.environ["JIANYANG_ARTIFACT_DIR"]
DOMAINS = set(os.environ["DOMAINS"].split(","))
MAX_TOKENS = int(os.environ.get("MAX_TOKENS", "1024"))
CONCURRENCY = int(os.environ.get("CONCURRENCY", "4"))
URL = f"http://127.0.0.1:{PORT}/v1/chat/completions"

items = [json.loads(line) for line in open(os.environ["PROMPTS"]) if line.strip()]
items = [item for item in items if item["domain"] in DOMAINS]
# LIMIT keeps the first N prompts of each domain, to bound long-reply domains.
if int(os.environ.get("LIMIT") or 0):
    kept = {}
    items = [item for item in items if kept.setdefault(item["domain"], []).append(item) or
             len(kept[item["domain"]]) <= int(os.environ["LIMIT"])]
# SHARD "i/n" keeps every n-th prompt starting at i, so n jobs split one domain.
if os.environ.get("SHARD"):
    shard, shards = (int(x) for x in os.environ["SHARD"].split("/"))
    items = items[shard::shards]


def answer(item):
    body = {"model": MODEL_ID, "messages": [{"role": "user", "content": item["prompt"]}],
            "max_tokens": MAX_TOKENS, "temperature": 0, "seed": 0,
            "chat_template_kwargs": {"enable_thinking": False}}
    request = urllib.request.Request(URL, data=json.dumps(body).encode(), headers={"content-type": "application/json"})
    for attempt in range(3):
        try:
            with urllib.request.urlopen(request, timeout=1800) as response:
                reply = json.load(response)
            choice = reply["choices"][0]
            return item, choice["message"].get("content") or "", choice.get("finish_reason"), \
                (reply.get("usage") or {}).get("completion_tokens", 0)
        except Exception as error:  # a reply is worth retrying; one prompt never fails the job
            if attempt == 2:
                print(f"{item['id']}: {error}", flush=True)
                return item, None, None, 0
            time.sleep(10)


suffix = f"-{os.environ['SHARD'].replace('/', 'of')}" if os.environ.get("SHARD") else ""
files = {domain: open(os.path.join(ART, f"train-{domain}{suffix}.jsonl"), "a") for domain in DOMAINS}
done = {domain: set() for domain in DOMAINS}
started, tokens, finished = time.monotonic(), 0, 0
with ThreadPoolExecutor(CONCURRENCY) as pool:
    for item, text, finish, out in pool.map(answer, items):
        finished += 1
        if not text:
            continue
        tokens += out
        files[item["domain"]].write(json.dumps({
            "id": item["id"], "finish_reason": finish,
            "conversations": [{"role": "user", "content": item["prompt"]},
                              {"role": "assistant", "content": text}]}) + "\n")
        files[item["domain"]].flush()
        done[item["domain"]].add(item["id"])
        if finished % 50 == 0:
            elapsed = time.monotonic() - started
            print(f"{finished}/{len(items)}  {tokens / elapsed:.0f} tok/s aggregate", flush=True)
print("DATAGEN " + json.dumps({d: len(ids) for d, ids in done.items()} | {"tokens": tokens,
                                "seconds": round(time.monotonic() - started)}), flush=True)
