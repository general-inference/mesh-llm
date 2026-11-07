"""Bench only: how well dumped prefill payloads compress (prints COMPRESS lines)."""
import os, shutil, struct, subprocess, sys, time, zlib
def comp(data, cmd):
    t=time.time(); out=subprocess.run(cmd, input=data, capture_output=True, check=True).stdout; dt=time.time()-t
    return len(out), dt
def report(name, data):
    n=len(data); rows=[]
    tools=[(l,c) for l,c in [("zstd-1",["zstd","-1","-c","-T1"]),("zstd-3",["zstd","-3","-c","-T1"]),("lz4",["lz4","-1","-c"])] if shutil.which(c[0])]
    for label, cmd in tools:
        size,dt=comp(data,cmd); rows.append(f"{label}: {n/size:.3f}x ({100*(1-size/n):.1f}% saved, {n/dt/1e6:.0f} MB/s)")
    t=time.time(); size=len(zlib.compress(data,1)); dt=time.time()-t
    rows.append(f"zlib-1: {n/size:.3f}x ({100*(1-size/n):.1f}% saved, {n/dt/1e6:.0f} MB/s)")
    print(f"COMPRESS {name} [{n/1e6:.1f} MB]: " + "; ".join(rows), flush=True)
d=sys.argv[1]
for f in sorted(os.listdir(d)):
    p=os.path.join(d,f)
    if f.startswith("dump-activation-t"):
        t=int(f[len("dump-activation-t"):-4]); raw=open(p,'rb').read()
        if t==0:
            vals=struct.unpack(f"<{len(raw)//4}f", raw); raw16=struct.pack(f"<{len(vals)}e", *vals)
            exact=sum(1 for v,w in zip(vals, struct.unpack(f"<{len(vals)}e", raw16)) if v!=w)
            print(f"COMPRESS f32 activations: {exact} of {len(vals)} values not exact in f16")
        else: raw16=raw
        report("activation f16", raw16)
        report("activation f16 byte-planes", raw16[1::2]+raw16[0::2])
    elif f=="dump-features.bin":
        raw=open(p,'rb').read(); report("features Q8", raw)
