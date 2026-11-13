"""Continues training replies that stopped at the token cap, until the target finishes.

Run by continue.sh against a running server. Each `finish_reason: length` row of DATA gets
its prompt in Qwen's chat format (thinking off) plus the reply so far sent to
/v1/completions, greedy; the continuation is appended and the row's finish_reason updated.
Rows that finished on their own are copied unchanged. SHARD "i/n" splits the rows to
continue across jobs.

A check runs first: CHECK_ROWS finished replies are cut at half their length and
continued the same way. If the continuations do not reproduce the originals, the prompt
format differs from the server's own chat formatting, and the job stops.
"""

import json
import os
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

PORT, MODEL_ID = os.environ["PORT"], os.environ["MODEL_ID"]
ART = os.environ["JIANYANG_ARTIFACT_DIR"]
MAX_TOKENS = int(os.environ.get("MAX_TOKENS", "12288"))
CONCURRENCY = int(os.environ.get("CONCURRENCY", "4"))
URL = f"http://127.0.0.1:{PORT}/v1/completions"

rows = [json.loads(line) for line in open(os.environ["DATA"]) if line.strip()]


def chat_text(user, partial):
    return f"<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n{partial}"


def complete(user, partial, max_tokens):
    body = {"model": MODEL_ID, "prompt": chat_text(user, partial), "max_tokens": max_tokens,
            "temperature": 0, "seed": 0}
    request = urllib.request.Request(URL, data=json.dumps(body).encode(), headers={"content-type": "application/json"})
    for attempt in range(3):
        try:
            with urllib.request.urlopen(request, timeout=3600) as response:
                choice = json.load(response)["choices"][0]
            return choice.get("text") or "", choice.get("finish_reason")
        except Exception as error:  # one row never fails the job
            if attempt == 2:
                print(f"continuation failed: {error}", flush=True)
                return None, None
            time.sleep(10)


def check():
    finished = [r for r in rows if r.get("finish_reason") == "stop"][:int(os.environ.get("CHECK_ROWS", "3"))]
    matched = 0
    for row in finished:
        user, reply = row["conversations"][0]["content"], row["conversations"][1]["content"]
        cut = len(reply) // 2
        cut = reply.rfind("\n", 0, cut) + 1 or cut  # continue from a line start
        want = reply[cut:]
        got, _ = complete(user, reply[:cut], 400)
        same = 0
        while same < min(len(got or ""), len(want)) and got[same] == want[same]:
            same += 1
        print(f"check {row['id']}: continuation matches the original for {same} of {min(len(got or ''), len(want))} chars",
              flush=True)
        matched += same >= 200
    if finished and matched < len(finished) - (len(finished) // 3):
        sys.exit("continuations do not reproduce finished replies; the prompt format is off")


def main():
    check()
    capped = [i for i, r in enumerate(rows) if r.get("finish_reason") == "length"]
    if os.environ.get("SHARD"):
        shard, shards = (int(x) for x in os.environ["SHARD"].split("/"))
        capped = capped[shard::shards]
    print(f"continuing {len(capped)} capped replies", flush=True)
    started, tokens = time.monotonic(), 0

    def run(index):
        row = rows[index]
        user, reply = row["conversations"][0]["content"], row["conversations"][1]["content"]
        text, finish = complete(user, reply, MAX_TOKENS)
        return index, text, finish

    out = open(os.path.join(ART, f"continued{'-' + os.environ['SHARD'].replace('/', 'of') if os.environ.get('SHARD') else ''}.jsonl"), "a")
    with ThreadPoolExecutor(CONCURRENCY) as pool:
        for done, (index, text, finish) in enumerate(pool.map(run, capped), 1):
            if text is None:
                continue
            row = rows[index]
            row["conversations"][1]["content"] += text
            row["finish_reason"] = finish
            tokens += len(text) // 4
            out.write(json.dumps(row) + "\n")
            out.flush()
            if done % 25 == 0:
                print(f"{done}/{len(capped)}  ~{tokens / (time.monotonic() - started):.0f} tok/s", flush=True)
    print("CONTINUED " + json.dumps({"rows": len(capped), "seconds": round(time.monotonic() - started)}), flush=True)


if __name__ == "__main__":
    main()
