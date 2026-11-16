# Serving recipes

These opt-in settings use the normal `skippy serve --settings` interface.
Supply the target and draft separately so paths remain local to the operator.
CLI and environment overrides still take precedence; see
[serving settings](../../../docs/SERVING_SETTINGS.md).

## Qwen3.6-35B-A3B with DFlash 2

```sh
skippy serve \
  --settings skippy/scripts/recipes/serving/qwen3.6-35b-a3b-dflash2.toml \
  --model "$TARGET_GGUF" \
  --draft-model-path "$DRAFT_GGUF"
```

Use a DFlash 2 draft trained for this target. The measured Q8_0 artifact is
[`ethanwtodd/Qwen3.6-35B-A3B-DFlash2-Q8_0-GGUF`](https://huggingface.co/ethanwtodd/Qwen3.6-35B-A3B-DFlash2-Q8_0-GGUF/tree/65aaab9513e9c6608ada7c7d16e6532c198a2296),
revision `65aaab9513e9c6608ada7c7d16e6532c198a2296`, file
`Qwen3.6-35B-A3B-DFlash2-Q8_0.gguf`, SHA-256
`986b8f24ac3469ed122946fb5be5b1fbe186467f2573f84e48bdbf7db278b327`.
It has eight block positions: one anchor and at most seven draft proposals.
The recipe keeps normal adaptive block sizing with seven as its ceiling and
disables native MTP while using DFlash.

Qualification used the original Unsloth
`Qwen3.6-35B-A3B-UD-Q4_K_M.gguf`, revision
`5bc3e238d916f48a861bac2f8a1990a0e9b7e98d`, SHA-256
`0b21525e972670ed59e1812e170b27c26355381f0656ecc4e25617ece7dac58b`.
On M5 Max, a paired comparison at six proposals improved fixed 128-token
completion throughput by 9.2% at concurrency one and 7.6% at concurrency two
versus the original DFlash draft. A separate paired comparison of seven versus
six proposals improved those rates by 1.2% and 2.1%. These separate comparisons
must not be multiplied into a combined speedup claim.

The broader dense/head/down Q4 recipe did not establish a DFlash 2 improvement
over DFlash 1. Draft acceptance and performance depend on target precision;
this preset is not a universal recommendation for every target quantization.
The measurements use two short coding prompts, greedy sampling, f16 KV, and
uncached requests. They do not establish performance on other workloads or
parity with BaseRT. See the
[decode qualification ledger](../../../docs/design/SKIPPY_METAL_DECODE_OPTIMIZATIONS.md).
