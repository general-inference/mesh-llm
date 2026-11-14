# HTML drafts by html_v2 format (served, greedy, 1024 tokens, 10 prompts per format)

Decode tok/s with the shipped DFlash2 draft and the change each fine-tuned draft makes, from
arms on one Spark per row group. v1: 600 Tailwind-only pages cut at 1024 tokens, 3 epochs.
v2: 999 complete pages in four formats, 2 epochs, 10% held out.

| Format | Shipped draft | v1 | v2, 1 epoch | v2, 2 epochs | v2 from Q4 features, 2 epochs |
|---|---:|---:|---:|---:|---:|
| Tailwind page | 91 | +57% | +35% | +48% | +44% |
| Plain-CSS page | 125 | +2% | +10% | +11% | +9% |
| Page edit | 125 | +1% | +9% | +14% | – |
| React + Tailwind component | 104 | −2% | +9% | +7% | – |

Jobs: job_1dj2g779mkt00bz, job_rnd5heaqatvve0r, job_qvdtsvzyebpt54p (Tailwind, plain CSS);
job_mgfhw08ttjkz80n, job_01csbp92y3z3teq (edit, React).
