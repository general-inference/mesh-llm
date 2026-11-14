# Quantization Recipes

This directory contains tensor-type override recipes for `skippy-quantize`.
Recipe files are intentionally comment-free because `skippy-quantize
validate-tensor-types` parses each whitespace-separated token as a
`PATTERN=TYPE` entry.

`PATTERN` is passed to llama.cpp as a regular expression and matched with
`std::regex_search`. First match wins, so put narrow overrides before broad
ones.

## GLM-5.2

Use `glm-5.2-q2-k-mtp-q8.tensor-types.txt` with base `--quant Q2_K` to
recreate the lab's comfort-fit GLM-5.2 profile:

- GLM attention projection tensors that were sensitive in local testing stay
  at `Q8_0`.
- Shared expert tensors stay at `Q4_K`.
- Native NextN/MTP tensors containing `.nextn.` stay at `Q8_0`.
- Routed expert gate/up tensors follow the `Q2_K` default.
- Routed expert down tensors follow llama.cpp's `Q2_K` profile fallback, which
  currently keeps them at `Q3_K` where required.

Use `glm-5.2-q2-k-routed-down-mtp-q8.tensor-types.txt` for the Phase E
routed-down experiment. It changes only decoder-layer routed down projections
to `Q2_K`:

```text
^blk\.([0-9]|[1-6][0-9]|7[0-7])\.ffn_down_exps\.weight$=Q2_K
```

The range deliberately excludes `blk.78`, the GLM-5.2 native NextN/MTP block.
That lets us test the measured decode throughput win from q2_K routed-down
without lowering the MTP block's sparse MLP down projection.

Validate recipes before launching an expensive quantization job:

```bash
skippy-quantize validate-tensor-types skippy/scripts/recipes/quantization/glm-5.2-q2-k-mtp-q8.tensor-types.txt
skippy-quantize validate-tensor-types skippy/scripts/recipes/quantization/glm-5.2-q2-k-routed-down-mtp-q8.tensor-types.txt
```

## Qwen3.6-35B-A3B decode profile

`qwen3.6-35b-a3b-ud-q4-k-m-decode.tensor-types.txt` is an explicit performance
profile for the unsloth `Qwen3.6-35B-A3B-UD-Q4_K_M.gguf` source with SHA-256
`0b21525e972670ed59e1812e170b27c26355381f0656ecc4e25617ece7dac58b`
(Hugging Face snapshot `5bc3e238d916f48a861bac2f8a1990a0e9b7e98d`). It is
specific to that source inventory; it is not a general conversion policy.

The recipe lists all 753 tensors explicitly. It changes 250 dense projection
weights and the output head to Q4_K, and 37 decoder expert-down tensors from
Q5_K to Q4_K. Blocks 34, 38, and 39 retain Q6_K expert-down weights. Routing
matrices, embeddings, norms, other expert tensors, and the entire block 40 MTP
sidecar retain their source types. The audited result changed exactly 288
tensors and preserved the other 465 byte-for-byte. Its size is 20,588,214,560
bytes and SHA-256 is
`2becfbe82ceb5862f90145c69b8545dcf30982aec2b81ce52ddeaec32648ecf9`.

Build `skippy-quantize` with `just skippy-quantize-build`. Set `SOURCE_GGUF`
to the absolute path of that source file, and `NATIVE_RUNTIME_LIBDIR` to the
locally built Metal runtime's `lib` directory (the macOS command below loads
`libllama-common`, `libllama`, and `libmtmd`). Run from the repository root. The
quantizer inventories its source directory, so isolate this GGUF from sibling
quantizations or multimodal projectors:

```bash
target/debug/skippy-quantize validate-tensor-types \
  skippy/scripts/recipes/quantization/qwen3.6-35b-a3b-ud-q4-k-m-decode.tensor-types.txt
mkdir -p ./qwen-decode-source
ln -s "$SOURCE_GGUF" ./qwen-decode-source/source.gguf
target/debug/skippy-quantize quantize --allow-requantize --no-stage-source \
  --native-runtime-library "$NATIVE_RUNTIME_LIBDIR/libllama-common.dylib" \
  --native-runtime-library "$NATIVE_RUNTIME_LIBDIR/libllama.dylib" \
  --native-runtime-library "$NATIVE_RUNTIME_LIBDIR/libmtmd.dylib" \
  --tensor-type-file skippy/scripts/recipes/quantization/qwen3.6-35b-a3b-ud-q4-k-m-decode.tensor-types.txt \
  ./qwen-decode-source/source.gguf ./Qwen3.6-35B-A3B-decode-Q4_K.gguf Q4_K_S 8
```

The product command emits `Qwen3.6-35B-A3B-decode-Q4_K-00001-of-00001.gguf`.
An end-to-end reproduction with the local packaged runtime produced the same
SHA-256 as the audited and benchmarked artifact above.

On the M5 Max fixed-prompt plain-decode comparison, adding the 37 expert-down
changes to the earlier dense/head Q4_K profile improved median throughput by
5.76% at concurrency one and 8.23% at concurrency two. Every measured candidate
batch exceeded every control batch; all 48 requests completed at 128 tokens.
This is a comparison against the dense/head profile, not the original model
or another engine. Generated text differs between quantizations.

On 128 WikiText-2 test contexts (32,640 scored positions), perplexity was
6.733014, compared with 6.732749 for the dense/head profile and 6.644816 for
the original-model compressed reference. Top-token agreement with the original
was 89.939%. These results describe one corpus and workload; they do not
establish general quality equivalence. See the
[Metal decode investigation](../../../docs/design/SKIPPY_METAL_DECODE_OPTIMIZATIONS.md)
for controls, limitations, and the earlier precision experiments. The profile
is opt-in and does not change model loading defaults.

## Qwen3.6 expert-down-only profile for DFlash 2

`qwen3.6-35b-a3b-ud-q4-k-m-expert-down.tensor-types.txt` uses the same pinned
Unsloth source as the decode profile above. Its complete 753-tensor inventory
changes only 37 Q5_K expert-down tensors to Q4_K, preserving the other 716
payloads byte-for-byte. Dense projections, the output head, routers, three
Q6_K expert-down tensors, and MTP retain their original precision.

Use the same isolated source and three native libraries as above, substituting
this recipe for `--tensor-type-file` and choosing a separate output filename.
The measured product-quantizer output is 21,421,873,440 bytes with SHA-256
`4ad842f4ed3ef1070d7818d95163e47402e7de17c79ecb85a0e272aab3ae52f5`.
It saves 1,241,513,984 bytes versus the source. This remains quantization from
an already compressed model and requires `--allow-requantize`.

With the pinned DFlash 2 draft and seven-proposal ceiling from the
[serving recipe](../serving/README.md), a fresh-process ABBA test of eight
short coding prompts at concurrency one measured 197.225 versus 186.864 tok/s
for the original target (+5.54%). All 128 requests, including warmups,
completed 128 tokens. Six prompts improved and two regressed; individual
median changes ranged from -15.31% to +17.45%. Generated text differs between
the targets and was stable within each prompt/target across measured repeats.
This is an opt-in workload-dependent tradeoff, not a default recommendation.

On the same 128-context WikiText-2 comparison, PPL was 6.657436 versus the
original compressed target's 6.644816; mean KL divergence was
0.008046 ± 0.000157 and top-token agreement was 96.317 ± 0.104%.
These results do not establish general quality equivalence or BaseRT parity.
See the decode investigation linked above for complete-configuration results,
controls, and limitations.
