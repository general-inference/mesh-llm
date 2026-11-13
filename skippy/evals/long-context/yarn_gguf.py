#!/usr/bin/env python3
"""Copies a GGUF with YaRN rope scaling written into its metadata.

    yarn_gguf.py SRC DST --factor 4 --original 262144

mesh-llm takes rope scaling from the GGUF (llama.cpp's `rope.scaling.*` keys); it has no
serve option for it. This writes `<arch>.rope.scaling.{type,factor,original_context_length}`
and raises `<arch>.context_length` to factor x original, replacing any existing values.
Tensor data is copied unchanged: tensor offsets are relative to the aligned data section,
so only the header grows.
"""

import argparse
import os
import shutil
import struct

U8, I8, U16, I16, U32, I32, F32, BOOL, STR, ARR, U64, I64, F64 = range(13)
SCALAR = {U8: "B", I8: "b", U16: "H", I16: "h", U32: "I", I32: "i", F32: "f", BOOL: "?", U64: "Q",
          I64: "q", F64: "d"}


class Reader:
    def __init__(self, f):
        self.f = f

    def take(self, fmt):
        size = struct.calcsize("<" + fmt)
        return struct.unpack("<" + fmt, self.f.read(size))[0]

    def string(self):
        return self.f.read(self.take("Q")).decode()

    def value(self, kind):
        if kind == STR:
            return self.string()
        if kind == ARR:
            item, n = self.take("I"), self.take("Q")
            return [self.value(item) for _ in range(n)]
        return self.take(SCALAR[kind])


def encode_kv(key, kind, value):
    raw = key.encode()
    out = struct.pack("<Q", len(raw)) + raw + struct.pack("<I", kind)
    if kind == STR:
        data = value.encode()
        return out + struct.pack("<Q", len(data)) + data
    return out + struct.pack("<" + SCALAR[kind], value)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("src")
    parser.add_argument("dst")
    parser.add_argument("--factor", type=float, required=True)
    parser.add_argument("--original", type=int, required=True)
    args = parser.parse_args()

    with open(args.src, "rb") as src:
        r = Reader(src)
        assert src.read(4) == b"GGUF", "not a GGUF file"
        version, n_tensors, n_kv = r.take("I"), r.take("Q"), r.take("Q")
        kept, arch, alignment = [], None, 32
        for _ in range(n_kv):
            start = src.tell()
            key = r.string()
            kind = r.take("I")
            value = r.value(kind)
            end = src.tell()
            if key == "general.architecture":
                arch = value
            if key == "general.alignment":
                alignment = value
            src.seek(start)
            kept.append((key, src.read(end - start)))
        assert arch, "no general.architecture"
        replaced = {f"{arch}.context_length", f"{arch}.rope.scaling.type", f"{arch}.rope.scaling.factor",
                    f"{arch}.rope.scaling.original_context_length", f"{arch}.rope.scaling.finetuned"}
        dropped = [key for key, _ in kept if key in replaced]
        kept = [raw for key, raw in kept if key not in replaced]
        added = [
            encode_kv(f"{arch}.context_length", U32, int(args.factor * args.original)),
            encode_kv(f"{arch}.rope.scaling.type", STR, "yarn"),
            encode_kv(f"{arch}.rope.scaling.factor", F32, args.factor),
            encode_kv(f"{arch}.rope.scaling.original_context_length", U32, args.original),
        ]
        info_start = src.tell()
        for _ in range(n_tensors):
            r.string()
            dims = r.take("I")
            src.read(8 * dims + 4 + 8)
        info_end = src.tell()
        src.seek(info_start)
        infos = src.read(info_end - info_start)
        data_start = -(-info_end // alignment) * alignment

        header = b"GGUF" + struct.pack("<IQQ", version, n_tensors, len(kept) + len(added))
        header += b"".join(kept) + b"".join(added) + infos
        header += b"\0" * (-len(header) % alignment)
        with open(args.dst + ".part", "wb") as dst:
            dst.write(header)
            src.seek(data_start)
            shutil.copyfileobj(src, dst, 64 << 20)
    os.replace(args.dst + ".part", args.dst)
    print(f"{args.dst}: {arch} yarn x{args.factor:g} over {args.original} "
          f"-> context_length {int(args.factor * args.original)} (replaced {dropped or 'nothing'})")


if __name__ == "__main__":
    main()
