#!/usr/bin/env python3
"""Offline suffix n-gram acceptance on the target's own greedy replies.

    ngram_sim.py --tokenizer tokenizer.json --workloads workloads.jsonl ARMS.jsonl ...

The served n-gram arm reports no draft counts, so this replays each reply token by token:
at every verify step, the longest suffix (ngram_min..ngram_max tokens) of prompt + reply
so far that occurred earlier proposes the tokens that followed it, up to --window; the
accepted prefix is what the reply actually continued with. Prompts are tokenized as raw
text (no chat template), which only shifts match positions, not what can be copied.
Reports mean accepted tokens per verify step and the share of steps that drafted at all.
"""

import argparse
import json
from collections import defaultdict

from tokenizers import Tokenizer


def simulate(context, reply, ngram_min, ngram_max, window):
    seq = list(context)
    index = defaultdict(list)  # ngram_min-gram -> positions just after it

    def add(upto):
        if upto >= ngram_min:
            index[tuple(seq[upto - ngram_min:upto])].append(upto)

    for i in range(1, len(seq) + 1):
        add(i)
    steps = drafted = accepted = 0
    i = 0
    while i < len(reply):
        steps += 1
        proposal = []
        key = tuple(seq[-ngram_min:]) if len(seq) >= ngram_min else None
        best_len, best_pos = 0, None
        for pos in reversed(index.get(key, [])[-64:]):
            if pos >= len(seq):
                continue
            length = ngram_min
            while length < ngram_max and pos - length - 1 >= 0 and \
                    seq[pos - length - 1] == seq[len(seq) - length - 1]:
                length += 1
            if length > best_len:
                best_len, best_pos = length, pos
        if best_pos is not None:
            proposal = seq[best_pos:best_pos + window]
        good = 0
        for token in proposal:
            if i + good < len(reply) and reply[i + good] == token:
                good += 1
            else:
                break
        if proposal:
            drafted += 1
        accepted += good
        # The verify step commits the accepted tokens plus the target's own next token.
        for token in reply[i:i + good + 1]:
            seq.append(token)
            add(len(seq))
        i += good + 1
    return steps, drafted, accepted


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--workloads", required=True)
    parser.add_argument("--agent-transcripts")
    parser.add_argument("--arm", default="off")
    parser.add_argument("--ngram-min", type=int, default=5)
    parser.add_argument("--ngram-max", type=int, default=32)
    parser.add_argument("--window", type=int, default=7)
    parser.add_argument("arms", nargs="+")
    args = parser.parse_args()
    tok = Tokenizer.from_file(args.tokenizer)
    prompts = {json.loads(l)["id"]: json.loads(l)["prompt"] for l in open(args.workloads) if l.strip()}
    if args.agent_transcripts:
        prompts.update({json.loads(l)["id"]: json.loads(l)["prompt"] for l in open(args.agent_transcripts)})
    # Greedy replies per group: the `off` arm's, else a speculative arm's (same text).
    replies = {}
    for path in args.arms:
        for line in open(path):
            arm = json.loads(line)
            if arm["arm"] not in (args.arm, f"agent_{args.arm}", "ngram", "agent_ngram"):
                continue
            for r in arm["results"]:
                if r["id"] not in prompts or not r.get("text"):
                    continue
                group = r["group"] if not arm["arm"].startswith("agent") else f"agent {r['group']}"
                preferred = not arm["arm"].endswith("ngram")
                have = replies.setdefault(group, {"preferred": preferred, "rows": []})
                if preferred and not have["preferred"]:
                    replies[group] = have = {"preferred": True, "rows": []}
                if preferred == have["preferred"]:
                    have["rows"].append(r)
    totals = defaultdict(lambda: [0, 0, 0, 0])
    for group, have in replies.items():
        for r in have["rows"]:
                ctx = tok.encode(prompts[r["id"]]).ids
                reply = tok.encode(r["text"], add_special_tokens=False).ids
                steps, drafted, accepted = simulate(ctx, reply, args.ngram_min, args.ngram_max, args.window)
                t = totals[group]
                t[0] += steps; t[1] += drafted; t[2] += accepted; t[3] += len(reply)
    print(f"| Workload | Reply tokens | Steps drafting | Accepted per step | Tokens per step |")
    print("|---|---:|---:|---:|---:|")
    for group, (steps, drafted, accepted, tokens) in sorted(totals.items()):
        print(f"| {group} | {tokens} | {drafted / steps:.0%} | {accepted / steps:.2f} | {tokens / steps:.2f} |")


if __name__ == "__main__":
    main()
