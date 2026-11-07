#!/usr/bin/env python3
"""glm_rename.py SRC_SNAPSHOT DST_FILE_DIR [paths...]

Rewrites meshllm/GLM-5.3-Flash-UD-Q4_K_XL-layers@d1b3be77 layer-package GGUFs the way the
au-mel-01 copy (manifest 1341078e) was rewritten to match Unsloth a38483c8: 'glm5next' keys
and architecture become 'glm5-next', the boolean glm5-next.attention.indexer.index_share_mtp=true
is inserted after general.repo_url, and skippy.package.tensor_offset moves by the header
growth. Tensor data is copied byte for byte. Without paths, rewrites every .gguf in the manifest.
"""
import json, os, struct, shutil, sys
OLD, NEW = b"glm5next", b"glm5-next"
INSERT_KEY = b"glm5-next.attention.indexer.index_share_mtp"
SZ = {0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i", 6: "f", 7: "?", 10: "Q", 11: "q", 12: "d"}

def parse(path):
    f = open(path, "rb")
    def rd(fmt): return struct.unpack("<" + fmt, f.read(struct.calcsize("<" + fmt)))
    def rstr(): (n,) = rd("Q"); return f.read(n)
    def rval(t):
        if t in SZ: return ("s", rd(SZ[t])[0])
        if t == 8: return ("str", rstr())
        if t == 9:
            (et,) = rd("I"); (n,) = rd("Q")
            if et in SZ: return ("arr", et, list(rd(SZ[et] * n)) if n else [])
            return ("arr", et, [rval(et) for _ in range(n)])
        raise ValueError(t)
    assert f.read(4) == b"GGUF"
    (ver,) = rd("I"); nt, nkv = rd("QQ")
    kvs = []
    for _ in range(nkv):
        k = rstr(); (t,) = rd("I"); kvs.append([k, t, rval(t)])
    tensors = []
    for _ in range(nt):
        name = rstr(); (nd,) = rd("I"); dims = rd("Q" * nd); tt, off = rd("IQ"); tensors.append((name, dims, tt, off))
    align = next((v[1] for k, t, v in kvs if k == b"general.alignment"), 32)
    hdr_end = f.tell(); data_start = (hdr_end + align - 1) // align * align
    f.close()
    return ver, kvs, tensors, align, data_start

def enc_val(t, v):
    if v[0] == "s": return struct.pack("<" + SZ[t], v[1])
    if v[0] == "str": return struct.pack("<Q", len(v[1])) + v[1]
    et, items = v[1], v[2]
    if et in SZ: return struct.pack("<IQ", et, len(items)) + (struct.pack("<" + SZ[et] * len(items), *items) if items else b"")
    return struct.pack("<IQ", et, len(items)) + b"".join(enc_val(et, x) for x in items)

def rewrite_kvs(kvs, offset_delta):
    out = []
    for k, t, v in kvs:
        if k.startswith(OLD + b"."): k = NEW + k[len(OLD):]
        if t == 8 and v[1] == OLD: v = ("str", NEW)
        if k == b"skippy.package.tensor_offset" and offset_delta:
            v = ("arr", v[1], [x + offset_delta for x in v[2]])
        out.append([k, t, v])
        # The new key closes the source model's keys, before the package's own skippy.* keys.
        if k == b"general.repo_url":
            out.append([INSERT_KEY, 7, ("s", True)])
    assert sum(x[0] == INSERT_KEY for x in out) == 1
    return out

def header(ver, kvs, tensors, align):
    parts = [b"GGUF", struct.pack("<I", ver), struct.pack("<QQ", len(tensors), len(kvs))]
    for k, t, v in kvs: parts.append(struct.pack("<Q", len(k)) + k + struct.pack("<I", t) + enc_val(t, v))
    for name, dims, tt, off in tensors:
        parts.append(struct.pack("<Q", len(name)) + name + struct.pack("<I", len(dims)) + struct.pack("<" + "Q" * len(dims), *dims) + struct.pack("<IQ", tt, off))
    h = b"".join(parts); return h + b"\0" * ((-len(h)) % align)

def main():
    snap, dst = sys.argv[1], sys.argv[2]
    paths = sys.argv[3:] or [e["path"] for e in json.load(open(os.path.join(snap, "model-package.json")))["artifact_catalog"]["entries"] if e["path"].endswith(".gguf")]
    os.makedirs(dst, exist_ok=True)
    for p in paths:
        src = os.path.join(snap, p)
        ver, kvs, tensors, align, data_start = parse(src)
        # The header grows by the same bytes in every part, so the data of every part moves by
        # one delta; compute it with offsets unchanged (they are fixed-width).
        delta = len(header(ver, rewrite_kvs(kvs, 0), tensors, align)) - data_start
        h = header(ver, rewrite_kvs(kvs, delta), tensors, align)
        out = os.path.join(dst, p); os.makedirs(os.path.dirname(out), exist_ok=True)
        with open(src, "rb") as f, open(out + ".part", "wb") as g:
            g.write(h); f.seek(data_start); shutil.copyfileobj(f, g, 64 << 20)
        os.replace(out + ".part", out)
        print(f"{p}: header {data_start} -> {len(h)} (+{delta})", flush=True)

main()
