#!/usr/bin/env python3
"""Which kinds of HTML tokens a draft gets right, from trace.sh's per-token traces.

    trace_table.py trace-generic.jsonl trace-<specialist>.jsonl [--by-set]

Each generated token is labelled by where it falls in the reply, using a small HTML/JSX
scanner over the reply text:
  class/style   inside a class=, className= or style= value
  attribute     any other attribute name or value
  tag           tag names and markup punctuation (<, </, >, =)
  text          visible text between tags
  css           inside <style>...</style>
  script        inside <script>...</script>, or code outside JSX in a component
  prose         text outside the reply's code block
and the table gives, per label, its share of tokens and the share of those tokens that came
from accepted drafts, for each trace. A token takes the label of its first character.
"""

import argparse
import json
import re
from collections import defaultdict

LABELS = ["class/style", "attribute", "tag", "text", "css", "script", "prose"]


def label_chars(text):
    """One label per character of `text`."""
    labels = []
    state, quote, attr, raw = "prose", None, "", None
    in_code = False
    i = 0
    while i < len(text):
        ch = text[i]
        if text.startswith("```", i):
            in_code = not in_code
            end = text.find("\n", i)
            end = len(text) if end < 0 else end + 1
            labels.extend(["prose"] * (end - i))
            i = end
            state = "text" if in_code else "prose"
            continue
        if not in_code:
            labels.append("prose")
            i += 1
            continue
        if raw:  # inside <style> or <script> until its closing tag
            if text.lower().startswith(f"</{raw}", i):
                raw = None
                state = "tag"
                labels.append("tag")
            else:
                labels.append("css" if raw == "style" else "script")
            i += 1
            continue
        if state in ("text", "script_code"):
            if ch == "<" and re.match(r"</?[A-Za-z!]", text[i:i + 3]):
                state = "tag"
                labels.append("tag")
            else:
                labels.append("text" if state == "text" else "script")
            i += 1
            continue
        if state == "tag":
            if ch == ">":
                labels.append("tag")
                tag = re.search(r"<\s*([A-Za-z]+)[^<]*$", text[:i])
                name = tag.group(1).lower() if tag else ""
                state = "text"
                if name in ("style", "script") and not text[:i].rstrip().endswith("/"):
                    raw = name
            elif ch in "\"'{":
                quote = "}" if ch == "{" else ch
                attr = re.search(r"([A-Za-z_:-]+)\s*=\s*$", text[:i])
                attr = attr.group(1).lower() if attr else ""
                state = "value"
                labels.append("tag")
            elif re.match(r"[A-Za-z_:-]", ch) and re.search(r"\s$", text[:i]):
                state = "attr_name"
                labels.append("attribute")
            else:
                labels.append("tag")
            i += 1
            continue
        if state == "attr_name":
            if re.match(r"[A-Za-z0-9_:-]", ch):
                labels.append("attribute")
            else:
                state = "tag"
                continue
            i += 1
            continue
        if state == "value":
            if ch == quote:
                state = "tag"
                labels.append("tag")
            else:
                labels.append("class/style" if attr in ("class", "classname", "style") else "attribute")
            i += 1
            continue
    return labels


def load(path):
    return {json.loads(line)["id"]: json.loads(line) for line in open(path)}


def summarise(rows):
    counts = defaultdict(lambda: [0, 0])
    for row in rows:
        text = "".join(row["pieces"])
        labels = label_chars(text)
        offset = 0
        for piece, drafted in zip(row["pieces"], row["from_draft"]):
            label = labels[offset] if offset < len(labels) and piece else "prose"
            counts[label][0] += 1
            counts[label][1] += int(drafted)
            offset += len(piece)
    return counts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("traces", nargs="+")
    parser.add_argument("--by-set", action="store_true", help="one table per eval set")
    args = parser.parse_args()
    traces = {path.rsplit("trace-", 1)[-1].removesuffix(".jsonl"): load(path) for path in args.traces}
    names = list(traces)
    sets = sorted({row["category"].split(":")[0] for t in traces.values() for row in t.values()}) \
        if args.by_set else [None]
    for subset in sets:
        print(f"\n{subset or 'all'}")
        print("| Token kind | Share of tokens | " + " | ".join(f"From draft: {n}" for n in names) + " |")
        print("|---|---:|" + "---:|" * len(names))
        sums = {n: summarise([r for r in traces[n].values()
                              if subset is None or r["category"].split(":")[0] == subset]) for n in names}
        total = sum(c[0] for c in sums[names[0]].values()) or 1
        for label in LABELS + ["all"]:
            rows = {n: (sum(c[0] for c in s.values()), sum(c[1] for c in s.values())) if label == "all"
                    else tuple(s.get(label, [0, 0])) for n, s in sums.items()}
            if rows[names[0]][0] == 0:
                continue
            share = rows[names[0]][0] / total
            cells = " | ".join(f"{d / t:.0%}" if t else "–" for t, d in rows.values())
            print(f"| {label} | {share:.0%} | {cells} |")


if __name__ == "__main__":
    main()
