#!/usr/bin/env python3
"""Adds the broader HTML sets to workloads.jsonl and train-prompts.jsonl.

    build_html_v2.py --src DIR [--train 1000] [--eval 10]

v1 HTML was one prompt style (a Tailwind page from a WebSight idea), so a draft trained on it
may only know that style. v2 spreads training over four formats, all from WebSight v0.2
(CC-BY-4.0):
  tailwind   a single-file Tailwind page from an idea (v1's format)
  plain_css  a single-file page with its own <style> block, no frameworks
  react      a React + TypeScript component with Tailwind classes
  edit       a WebSight page and a change to make; reply with the whole page
Eval rows go to workloads.jsonl as domain `html_v2` (--eval per format) and `html_ood`:
40 real edits of HTML, JSX and Vue files from bigcode/commitpackft (MIT), a source no
training prompt comes from. Training rows go to train-prompts.jsonl as domain `html_v2`,
never sharing an idea or page with any eval row (v1's `html` included). Existing rows of
both files are left as they are.

`--src` holds commitpackft's html.jsonl, jsx.jsonl and vue.jsonl.
"""

import argparse
import hashlib
import json
import random
from pathlib import Path

from build_corpora import MAX_FILE_CHARS, rows

HERE = Path(__file__).resolve().parent
FORMATS = ("tailwind", "plain_css", "react", "edit")
EDITS = [
    "Add a dark mode toggle button to the header that switches the page between light and dark colours.",
    "Add a contact form section (name, email, message, submit button) just before the footer.",
    "Make the navigation bar stick to the top of the page while scrolling.",
    "Add an FAQ section with four questions and answers after the main content.",
    "Change the colour scheme from its current colours to shades of green.",
    "Add a testimonials section with three customer quotes.",
    "Make the layout work on phones: stack the columns and add a hamburger menu for the navigation.",
    "Add a pricing section with three plans in cards.",
    "Translate all visible text on the page into French.",
    "Add a newsletter signup box to the footer.",
    "Replace the main heading and its paragraph with copy for a product launch announcement.",
    "Add a back-to-top button that appears at the bottom right of the page.",
]


def stable_id(*parts):
    return hashlib.sha256("\x00".join(parts).encode()).hexdigest()[:12]


def prompt_for(fmt, idea, page, edit):
    if fmt == "tailwind":
        return ("Write a complete single-file HTML page using Tailwind CSS from the CDN for this website. "
                f"Reply with only the HTML.\n\n{idea}")
    if fmt == "plain_css":
        return ("Write a complete single-file HTML page for this website. Put all styling in one <style> block; "
                f"use no CSS frameworks and no external files. Reply with only the HTML.\n\n{idea}")
    if fmt == "react":
        return ("Write one React component in TypeScript, styled with Tailwind classes, that implements this "
                f"website's page. Reply with only the code in one tsx code block.\n\n{idea}")
    return (f"Here is a web page:\n\n```html\n{page}\n```\n\n{edit}\n\n"
            "Reply with the complete updated page in one html code block and nothing else.")


def websight_pages(need, rng):
    """WebSight rows with both an idea and a page short enough to edit."""
    out, seen = [], set()
    for offset in rng.sample(range(0, 1_900_000, 100), need // 40 + 20):
        for row in rows("HuggingFaceM4/WebSight", "v0.2", "train", offset, 100):
            idea, page = (row.get("llm_generated_idea") or "").strip(), (row.get("text") or "").strip()
            if not idea or not page or len(page) > MAX_FILE_CHARS or idea in seen:
                continue
            seen.add(idea)
            out.append((idea, page))
        if len(out) >= need:
            break
    return out


def ood_edits(src, n, rng):
    picked = []
    for lang, fence, count in (("html", "html", n // 2), ("jsx", "jsx", n // 4), ("vue", "vue", n - n // 2 - n // 4)):
        rows_ = []
        for line in open(Path(src) / f"{lang}.jsonl"):
            row = json.loads(line)
            old, new = row["old_contents"], row["new_contents"]
            if old.strip() and old != new and len(old) <= MAX_FILE_CHARS and len(new) <= MAX_FILE_CHARS:
                rows_.append(row)
        rng.shuffle(rows_)
        for row in rows_[:count]:
            picked.append({"id": f"htmlood-{row['commit'][:12]}", "source": f"bigcode/commitpackft:{lang}",
                           "prompt": (f"Apply this change to `{row['old_file']}`: {row['subject']}\n\n"
                                      "Reply with the complete updated file in one code block and nothing else."
                                      f"\n\n```{fence}\n{row['old_contents']}```")})
    return picked


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", required=True)
    parser.add_argument("--train", type=int, default=1000)
    parser.add_argument("--eval", type=int, default=10)
    args = parser.parse_args()
    rng = random.Random(20261114)
    workloads = [json.loads(line) for line in open(HERE / "workloads.jsonl") if line.strip()]
    if any(row["domain"] in ("html_v2", "html_ood") for row in workloads):
        raise SystemExit("workloads.jsonl already has the v2 HTML rows")
    # v1's eval ideas stay out of everything v2 trains on, and v1's training ideas out of
    # v2's eval, so the v1 draft is not measured on pages it trained on.
    v1_eval = {row["prompt"].rsplit("\n\n", 1)[-1] for row in workloads if row["domain"] == "html"}
    v1_train_path = HERE / "data" / "train-html.jsonl"
    v1_train = {json.loads(line)["conversations"][0]["content"].rsplit("\n\n", 1)[-1]
                for line in open(v1_train_path)} if v1_train_path.exists() else set()
    per_format = args.train // len(FORMATS) + args.eval
    pages = [p for p in websight_pages(per_format * len(FORMATS) + 50, rng) if p[0] not in v1_eval]
    rng.shuffle(pages)
    eval_rows, train_rows = [], []
    for index, fmt in enumerate(FORMATS):
        chunk = pages[index * per_format:(index + 1) * per_format]
        held = 0
        for idea, page in chunk:
            edit = EDITS[rng.randrange(len(EDITS))]
            row = {"domain": "html_v2", "format": fmt, "id": f"htmlv2-{fmt}-{stable_id(idea)}",
                   "source": "HuggingFaceM4/WebSight@v0.2", "prompt": prompt_for(fmt, idea, page, edit)}
            if held < args.eval and idea not in v1_train:
                held += 1
                eval_rows.append(dict(row, split="eval"))
            else:
                train_rows.append(dict(row, split="train"))
    eval_rows += [dict(row, domain="html_ood", split="eval") for row in ood_edits(args.src, 40, rng)]
    with open(HERE / "workloads.jsonl", "a") as f:
        for row in eval_rows:
            f.write(json.dumps(row) + "\n")
    with open(HERE / "train-prompts.jsonl", "a") as f:
        for row in train_rows:
            f.write(json.dumps(row) + "\n")
    count = lambda rs, key: {k: sum(1 for r in rs if r.get(key) == k) for k in sorted({r.get(key) for r in rs})}  # noqa: E731
    print("eval", count(eval_rows, "domain"), count([r for r in eval_rows if r["domain"] == "html_v2"], "format"))
    print("train", len(train_rows), count(train_rows, "format"))


if __name__ == "__main__":
    main()
