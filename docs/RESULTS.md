# Results

Generated 2026-08-17 02:32 UTC by `python -m harness.sweep`. Do not edit by hand.

## Environment

| | |
| --- | --- |
| Python | 3.11.16 |
| PyTorch | 2.13.0+cpu |
| Triton | 3.7.1 |
| Execution | Triton CPU interpreter (TRITON_INTERPRET=1) |
| GPU | none (CPU only) |
| Measured peak bandwidth | not measured |

## Correctness

Kernels are compared against naive PyTorch references. Tolerances are derived
from dtype machine epsilon in `harness/config.py`, not tuned to make tests
pass. `fusion_not_less_accurate` requires the fused kernel to be at least as
close to an fp64 ground truth as the unfused two-kernel sequence.

| shape | dtype | check | max abs err | max rel err | rtol | atol | result |
| --- | --- | --- | ---: | ---: | ---: | ---: | --- |
| 128x512 | float32 | rmsnorm | 1.431e-06 | 3.184e-07 | 1e-05 | 1e-06 | PASS |
| 128x512 | float32 | softmax | 7.451e-09 | 3.463e-07 | 1e-05 | 1e-06 | PASS |
| 128x512 | float32 | fused | 3.278e-07 | 1.857e-06 | 1e-05 | 1e-06 | PASS |
| 128x512 | float32 | fusion_not_less_accurate | n/a | 1.584e-06 | n/a | n/a | PASS |
| 256x1024 | float32 | rmsnorm | 1.907e-06 | 3.216e-07 | 1e-05 | 1e-06 | PASS |
| 256x1024 | float32 | softmax | 5.588e-09 | 4.075e-07 | 1e-05 | 1e-06 | PASS |
| 256x1024 | float32 | fused | 2.682e-07 | 2.822e-06 | 1e-05 | 1e-06 | PASS |
| 256x1024 | float32 | fusion_not_less_accurate | n/a | 1.786e-06 | n/a | n/a | PASS |
| 512x2048 | float32 | rmsnorm | 1.907e-06 | 3.349e-07 | 1e-05 | 1e-06 | PASS |
| 512x2048 | float32 | softmax | 3.725e-09 | 4.328e-07 | 1e-05 | 1e-06 | PASS |
| 512x2048 | float32 | fused | 3.576e-07 | 3.616e-06 | 1e-05 | 1e-06 | PASS |
| 512x2048 | float32 | fusion_not_less_accurate | n/a | 3.248e-06 | n/a | n/a | PASS |
| 1024x4096 | float32 | rmsnorm | 1.907e-06 | 3.475e-07 | 1e-05 | 1e-06 | PASS |
| 1024x4096 | float32 | softmax | 2.794e-09 | 4.420e-07 | 1e-05 | 1e-06 | PASS |
| 1024x4096 | float32 | fused | 5.066e-07 | 3.971e-06 | 1e-05 | 1e-06 | PASS |
| 1024x4096 | float32 | fusion_not_less_accurate | n/a | 2.170e-06 | n/a | n/a | PASS |
| 128x512 | bfloat16 | rmsnorm | 6.250e-02 | 7.752e-03 | 0.02 | 1e-06 | PASS |
| 128x512 | bfloat16 | softmax | 4.883e-04 | 7.752e-03 | 0.02 | 1e-06 | PASS |
| 128x512 | bfloat16 | fused | 3.906e-03 | 7.752e-03 | 0.02 | 1e-06 | PASS |
| 128x512 | bfloat16 | fusion_not_less_accurate | n/a | 7.746e-03 | n/a | n/a | PASS |
| 256x1024 | bfloat16 | rmsnorm | 6.250e-02 | 7.752e-03 | 0.02 | 1e-06 | PASS |
| 256x1024 | bfloat16 | softmax | 2.441e-04 | 7.752e-03 | 0.02 | 1e-06 | PASS |
| 256x1024 | bfloat16 | fused | 3.906e-03 | 7.752e-03 | 0.02 | 1e-06 | PASS |
| 256x1024 | bfloat16 | fusion_not_less_accurate | n/a | 7.752e-03 | n/a | n/a | PASS |
| 512x2048 | bfloat16 | rmsnorm | 6.250e-02 | 7.752e-03 | 0.02 | 1e-06 | PASS |
| 512x2048 | bfloat16 | softmax | 1.221e-04 | 7.752e-03 | 0.02 | 1e-06 | PASS |
| 512x2048 | bfloat16 | fused | 3.906e-03 | 7.752e-03 | 0.02 | 1e-06 | PASS |
| 512x2048 | bfloat16 | fusion_not_less_accurate | n/a | 7.752e-03 | n/a | n/a | PASS |
| 1024x4096 | bfloat16 | rmsnorm | 6.250e-02 | 7.752e-03 | 0.02 | 1e-06 | PASS |
| 1024x4096 | bfloat16 | softmax | 1.221e-04 | 7.752e-03 | 0.02 | 1e-06 | PASS |
| 1024x4096 | bfloat16 | fused | 3.906e-03 | 7.752e-03 | 0.02 | 1e-06 | PASS |
| 1024x4096 | bfloat16 | fusion_not_less_accurate | n/a | 7.752e-03 | n/a | n/a | PASS |

**Summary: 32 of 32 asserted checks passed.**

### Measurements without pass/fail semantics

`fused_vs_unfused` is a divergence, not an error: it records how far apart the
two pipelines land. In bf16 they diverge substantially, and the two accuracy
rows show why -- the unfused sequence rounds its intermediate to bf16 before
softmax sees it, while the fused kernel keeps it in fp32 registers. Lower is
better in the two `*_accuracy_fp64` rows.

| shape | dtype | measurement | max abs err | max rel err |
| --- | --- | --- | ---: | ---: |
| 128x512 | float32 | fused_vs_unfused | 0.000e+00 | 0.000e+00 |
| 128x512 | float32 | fused_accuracy_fp64 | 1.605e-07 | 1.584e-06 |
| 128x512 | float32 | unfused_accuracy_fp64 | 1.605e-07 | 1.584e-06 |
| 256x1024 | float32 | fused_vs_unfused | 0.000e+00 | 0.000e+00 |
| 256x1024 | float32 | fused_accuracy_fp64 | 2.086e-07 | 1.786e-06 |
| 256x1024 | float32 | unfused_accuracy_fp64 | 2.086e-07 | 1.786e-06 |
| 512x2048 | float32 | fused_vs_unfused | 0.000e+00 | 0.000e+00 |
| 512x2048 | float32 | fused_accuracy_fp64 | 2.678e-07 | 3.248e-06 |
| 512x2048 | float32 | unfused_accuracy_fp64 | 2.678e-07 | 3.248e-06 |
| 1024x4096 | float32 | fused_vs_unfused | 0.000e+00 | 0.000e+00 |
| 1024x4096 | float32 | fused_accuracy_fp64 | 3.174e-07 | 2.170e-06 |
| 1024x4096 | float32 | unfused_accuracy_fp64 | 3.174e-07 | 2.170e-06 |
| 128x512 | bfloat16 | fused_vs_unfused | 5.859e-03 | 6.383e-02 |
| 128x512 | bfloat16 | fused_accuracy_fp64 | 3.905e-03 | 7.746e-03 |
| 128x512 | bfloat16 | unfused_accuracy_fp64 | 7.490e-03 | 6.617e-02 |
| 256x1024 | bfloat16 | fused_vs_unfused | 1.562e-02 | 7.600e-02 |
| 256x1024 | bfloat16 | fused_accuracy_fp64 | 3.666e-03 | 7.752e-03 |
| 256x1024 | bfloat16 | unfused_accuracy_fp64 | 1.656e-02 | 7.782e-02 |
| 512x2048 | bfloat16 | fused_vs_unfused | 1.172e-02 | 7.203e-02 |
| 512x2048 | bfloat16 | fused_accuracy_fp64 | 3.642e-03 | 7.752e-03 |
| 512x2048 | bfloat16 | unfused_accuracy_fp64 | 1.476e-02 | 7.689e-02 |
| 1024x4096 | bfloat16 | fused_vs_unfused | 1.367e-02 | 8.333e-02 |
| 1024x4096 | bfloat16 | fused_accuracy_fp64 | 3.881e-03 | 7.752e-03 |
| 1024x4096 | bfloat16 | unfused_accuracy_fp64 | 1.491e-02 | 9.082e-02 |

## Memory traffic (analytical)

Derived from shapes and dtype sizes only. Nothing here is measured; the
arithmetic is written out in the module docstring of `harness/memory_model.py`
so it can be verified by hand.

The `reduction` column assumes the N-element weight vector is fetched from HBM
once and served from cache afterwards. The `cold W` column assumes it is
re-fetched for every row. Reality sits near the first bound for these shapes,
and the second is shown so the claim is bracketed rather than asserted.

| shape | dtype | eager (MiB) | unfused (MiB) | fused (MiB) | reduction | reduction (cold W) | traffic ratio vs eager |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 128x512 | float32 | 3.76 | 1.00 | 0.50 | 49.90% | 40.00% | 7.48x |
| 256x1024 | float32 | 15.01 | 4.00 | 2.00 | 49.95% | 40.00% | 7.49x |
| 512x2048 | float32 | 60.02 | 16.01 | 8.01 | 49.98% | 40.00% | 7.50x |
| 1024x4096 | float32 | 240.05 | 64.02 | 32.02 | 49.99% | 40.00% | 7.50x |
| 128x512 | bfloat16 | 1.88 | 0.50 | 0.25 | 49.90% | 40.00% | 7.48x |
| 256x1024 | bfloat16 | 7.51 | 2.00 | 1.00 | 49.95% | 40.00% | 7.49x |
| 512x2048 | bfloat16 | 30.01 | 8.00 | 4.00 | 49.98% | 40.00% | 7.50x |
| 1024x4096 | bfloat16 | 120.02 | 32.01 | 16.01 | 49.99% | 40.00% | 7.50x |

The saving is exactly `2*M*N` elements in every case: one write of the
intermediate and one read of it back. As M grows, the N-element weight term
becomes negligible and the reduction tends to 50%.

## Wall-clock performance

**not measured** -- no GPU was available when this report was generated. Wall-clock timing is only collected on real hardware; see `notebooks/colab_benchmark.ipynb`.

## What these results do and do not establish

Established:

- The three kernels compute what the PyTorch references compute, across four
  shapes and two dtypes, within tolerances derived from dtype epsilon.
- Fusing RMSNorm into softmax does not cost accuracy. In bf16 it measurably
  improves it, because the intermediate is never rounded to storage precision.
- The fused kernel moves half as many bytes to and from HBM as the unfused
  pair, and roughly 7.5x fewer than naive eager PyTorch. This is arithmetic
  from shapes and dtype sizes, not a measurement.

Not established without a GPU run:

- Any wall-clock speedup. The correctness figures above come from the Triton
  CPU interpreter, which executes kernels sequentially in Python via NumPy and
  models neither the memory hierarchy nor occupancy. No timing taken there
  would mean anything, so none is reported.
- Whether the predicted traffic reduction actually converts into a
  proportional speedup. That depends on how close each variant runs to the
  bandwidth roofline, which requires hardware to determine.
