"""Aggregates an nsys cuda_gpu_trace CSV by stream: total GPU time per stream and its top kernels.

Each llama context (target, DFlash draft) launches on its own stream, so this splits a
speculative step's GPU time between them.
"""
import collections, csv, sys

rows = csv.DictReader(l for l in open(sys.argv[1]) if "," in l)
total = collections.Counter(); kern = collections.defaultdict(collections.Counter); calls = collections.defaultdict(collections.Counter)
for r in rows:
    name = r.get("Name", "")
    if name.startswith("[CUDA"):
        name = name  # memcpy/memset rows keep their label
    strm = r.get("Strm") or r.get("Stream") or "?"
    dur = float(r.get("Duration (ns)") or r.get("Duration") or 0)
    total[strm] += dur; kern[strm][name[:110]] += dur; calls[strm][name[:110]] += 1
grand = sum(total.values()) or 1
for strm, t in total.most_common(6):
    print(f"STREAM {strm}: {t/1e9:.3f} s ({100*t/grand:.1f}%)")
    for name, d in kern[strm].most_common(12):
        print(f"   {100*d/t:5.1f}%  n={calls[strm][name]:>6}  {name}")
