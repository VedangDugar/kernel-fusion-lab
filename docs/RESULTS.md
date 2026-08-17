# Results

Generated 2026-08-17 03:43 UTC by `python -m harness.sweep`. Do not edit by hand.

## Environment

| | |
| --- | --- |
| Python | 3.12.13 |
| PyTorch | 2.11.0+cu128 |
| Triton | 3.6.0 |
| Execution | compiled (native) |
| Correctness ran on | cuda |
| GPU | Tesla T4 |
| Measured peak bandwidth | 238.9 GB/s (measured by device-to-device copy) |

## Correctness

Kernels are compared against naive PyTorch references. Tolerances are derived
from dtype machine epsilon in `harness/config.py`, not tuned to make tests
pass. `fusion_not_less_accurate` requires the fused kernel to be at least as
close to an fp64 ground truth as the unfused two-kernel sequence.

| shape | dtype | check | max abs err | max rel err | rtol | atol | result |
| --- | --- | --- | ---: | ---: | ---: | ---: | --- |
| 128x512 | float32 | rmsnorm | 9.537e-07 | 3.116e-07 | 1e-05 | 1e-06 | PASS |
| 128x512 | float32 | softmax | 5.588e-09 | 5.297e-07 | 1e-05 | 1e-06 | PASS |
| 128x512 | float32 | fused | 1.788e-07 | 1.863e-06 | 1e-05 | 1e-06 | PASS |
| 128x512 | float32 | fusion_not_less_accurate | n/a | 1.664e-06 | n/a | n/a | PASS |
| 256x1024 | float32 | rmsnorm | 1.431e-06 | 3.778e-07 | 1e-05 | 1e-06 | PASS |
| 256x1024 | float32 | softmax | 3.725e-09 | 5.720e-07 | 1e-05 | 1e-06 | PASS |
| 256x1024 | float32 | fused | 2.980e-07 | 2.706e-06 | 1e-05 | 1e-06 | PASS |
| 256x1024 | float32 | fusion_not_less_accurate | n/a | 2.619e-06 | n/a | n/a | PASS |
| 512x2048 | float32 | rmsnorm | 1.907e-06 | 3.456e-07 | 1e-05 | 1e-06 | PASS |
| 512x2048 | float32 | softmax | 3.725e-09 | 6.456e-07 | 1e-05 | 1e-06 | PASS |
| 512x2048 | float32 | fused | 4.172e-07 | 3.673e-06 | 1e-05 | 1e-06 | PASS |
| 512x2048 | float32 | fusion_not_less_accurate | n/a | 2.999e-06 | n/a | n/a | PASS |
| 1024x4096 | float32 | rmsnorm | 2.861e-06 | 4.328e-07 | 1e-05 | 1e-06 | PASS |
| 1024x4096 | float32 | softmax | 1.863e-09 | 6.743e-07 | 1e-05 | 1e-06 | PASS |
| 1024x4096 | float32 | fused | 5.960e-07 | 3.870e-06 | 1e-05 | 1e-06 | PASS |
| 1024x4096 | float32 | fusion_not_less_accurate | n/a | 2.986e-06 | n/a | n/a | PASS |
| 128x512 | bfloat16 | rmsnorm | 0.000e+00 | 0.000e+00 | 0.02 | 1e-06 | PASS |
| 128x512 | bfloat16 | softmax | 0.000e+00 | 0.000e+00 | 0.02 | 1e-06 | PASS |
| 128x512 | bfloat16 | fused | 7.629e-06 | 6.173e-03 | 0.02 | 1e-06 | PASS |
| 128x512 | bfloat16 | fusion_not_less_accurate | n/a | 3.886e-03 | n/a | n/a | PASS |
| 256x1024 | bfloat16 | rmsnorm | 2.441e-04 | 5.000e-03 | 0.02 | 1e-06 | PASS |
| 256x1024 | bfloat16 | softmax | 3.815e-06 | 6.061e-03 | 0.02 | 1e-06 | PASS |
| 256x1024 | bfloat16 | fused | 3.815e-06 | 7.692e-03 | 0.02 | 1e-06 | PASS |
| 256x1024 | bfloat16 | fusion_not_less_accurate | n/a | 3.891e-03 | n/a | n/a | PASS |
| 512x2048 | bfloat16 | rmsnorm | 7.812e-03 | 6.993e-03 | 0.02 | 1e-06 | PASS |
| 512x2048 | bfloat16 | softmax | 1.526e-05 | 6.849e-03 | 0.02 | 1e-06 | PASS |
| 512x2048 | bfloat16 | fused | 7.629e-06 | 7.692e-03 | 0.02 | 1e-06 | PASS |
| 512x2048 | bfloat16 | fusion_not_less_accurate | n/a | 3.891e-03 | n/a | n/a | PASS |
| 1024x4096 | bfloat16 | rmsnorm | 3.125e-02 | 7.692e-03 | 0.02 | 1e-06 | PASS |
| 1024x4096 | bfloat16 | softmax | 3.815e-06 | 7.519e-03 | 0.02 | 1e-06 | PASS |
| 1024x4096 | bfloat16 | fused | 1.526e-05 | 7.634e-03 | 0.02 | 1e-06 | PASS |
| 1024x4096 | bfloat16 | fusion_not_less_accurate | n/a | 3.891e-03 | n/a | n/a | PASS |

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
| 128x512 | float32 | fused_accuracy_fp64 | 2.437e-07 | 1.664e-06 |
| 128x512 | float32 | unfused_accuracy_fp64 | 2.437e-07 | 1.664e-06 |
| 256x1024 | float32 | fused_vs_unfused | 0.000e+00 | 0.000e+00 |
| 256x1024 | float32 | fused_accuracy_fp64 | 2.546e-07 | 2.619e-06 |
| 256x1024 | float32 | unfused_accuracy_fp64 | 2.546e-07 | 2.619e-06 |
| 512x2048 | float32 | fused_vs_unfused | 0.000e+00 | 0.000e+00 |
| 512x2048 | float32 | fused_accuracy_fp64 | 3.156e-07 | 2.999e-06 |
| 512x2048 | float32 | unfused_accuracy_fp64 | 3.156e-07 | 2.999e-06 |
| 1024x4096 | float32 | fused_vs_unfused | 0.000e+00 | 0.000e+00 |
| 1024x4096 | float32 | fused_accuracy_fp64 | 3.929e-07 | 2.986e-06 |
| 1024x4096 | float32 | unfused_accuracy_fp64 | 3.929e-07 | 2.986e-06 |
| 128x512 | bfloat16 | fused_vs_unfused | 3.906e-03 | 3.623e-02 |
| 128x512 | bfloat16 | fused_accuracy_fp64 | 1.914e-03 | 3.886e-03 |
| 128x512 | bfloat16 | unfused_accuracy_fp64 | 4.296e-03 | 3.729e-02 |
| 256x1024 | bfloat16 | fused_vs_unfused | 7.812e-03 | 3.623e-02 |
| 256x1024 | bfloat16 | fused_accuracy_fp64 | 1.790e-03 | 3.891e-03 |
| 256x1024 | bfloat16 | unfused_accuracy_fp64 | 7.351e-03 | 3.496e-02 |
| 512x2048 | bfloat16 | fused_vs_unfused | 7.812e-03 | 4.651e-02 |
| 512x2048 | bfloat16 | fused_accuracy_fp64 | 1.877e-03 | 3.891e-03 |
| 512x2048 | bfloat16 | unfused_accuracy_fp64 | 7.141e-03 | 4.328e-02 |
| 1024x4096 | bfloat16 | fused_vs_unfused | 7.812e-03 | 4.525e-02 |
| 1024x4096 | bfloat16 | fused_accuracy_fp64 | 1.865e-03 | 3.891e-03 |
| 1024x4096 | bfloat16 | unfused_accuracy_fp64 | 8.104e-03 | 4.701e-02 |

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

Median of the runtime distribution with 20th/80th percentiles, via
`triton.testing.do_bench`. Percentages are against *measured* achievable
bandwidth, not the datasheet figure.

`GB/s (model-implied)` divides the analytical byte count by the measured time.
That is only meaningful where the byte count is actually known, which is the
two Triton kernels, since their loads and stores are visible in the source.
`torch.compile` decides for itself how much to fuse, so the column is left
blank for the PyTorch providers rather than filled with a number derived from
an assumption that does not hold.

Where a Triton row exceeds 100% of measured peak, the kernel is not breaking
physics -- it is evidence that some of the traffic the model attributes to HBM
was served from L2 instead. See the next section.

| shape | dtype | provider | median (ms) | p20-p80 (ms) | GB/s (model-implied) | % of measured peak |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| 128x512 | float32 | triton_fused | 0.0092 | 0.0082-0.0102 | 57.5 | 24.1% |
| 128x512 | float32 | triton_unfused | 0.0138 | 0.0131-0.0142 | 76.4 | 32.0% |
| 128x512 | float32 | torch_eager | 0.0432 | 0.0430-0.0703 | traffic not known | traffic not known |
| 128x512 | float32 | torch_compile | 0.0074 | 0.0067-0.0080 | traffic not known | traffic not known |
| 256x1024 | float32 | triton_fused | 0.0131 | 0.0124-0.0139 | 160.5 | 67.2% |
| 256x1024 | float32 | triton_unfused | 0.0160 | 0.0152-0.0164 | 261.9 | 109.6% |
| 256x1024 | float32 | torch_eager | 0.0608 | 0.0600-0.0613 | traffic not known | traffic not known |
| 256x1024 | float32 | torch_compile | 0.0149 | 0.0144-0.0156 | traffic not known | traffic not known |
| 512x2048 | float32 | triton_fused | 0.0410 | 0.0410-0.0413 | 205.0 | 85.8% |
| 512x2048 | float32 | triton_unfused | 0.0752 | 0.0742-0.0757 | 223.1 | 93.4% |
| 512x2048 | float32 | torch_eager | 0.2600 | 0.2590-0.2602 | traffic not known | traffic not known |
| 512x2048 | float32 | torch_compile | 0.0430 | 0.0429-0.0436 | traffic not known | traffic not known |
| 1024x4096 | float32 | triton_fused | 0.1507 | 0.1497-0.1516 | 222.7 | 93.2% |
| 1024x4096 | float32 | triton_unfused | 0.2955 | 0.2948-0.2967 | 227.2 | 95.1% |
| 1024x4096 | float32 | torch_eager | 1.0756 | 1.0751-1.0767 | traffic not known | traffic not known |
| 1024x4096 | float32 | torch_compile | 0.1720 | 0.1711-0.1732 | traffic not known | traffic not known |
| 128x512 | bfloat16 | triton_fused | 0.0060 | 0.0058-0.0061 | 44.0 | 18.4% |
| 128x512 | bfloat16 | triton_unfused | 0.0079 | 0.0072-0.0082 | 66.2 | 27.7% |
| 128x512 | bfloat16 | torch_eager | 0.0576 | 0.0573-0.0584 | traffic not known | traffic not known |
| 128x512 | bfloat16 | torch_compile (see note) | 0.0512 | 0.0512-0.0516 | traffic not known | traffic not known |
| 256x1024 | bfloat16 | triton_fused | 0.0090 | 0.0086-0.0096 | 116.4 | 48.7% |
| 256x1024 | bfloat16 | triton_unfused | 0.0119 | 0.0110-0.0122 | 177.1 | 74.1% |
| 256x1024 | bfloat16 | torch_eager | 0.0793 | 0.0785-0.0798 | traffic not known | traffic not known |
| 256x1024 | bfloat16 | torch_compile (see note) | 0.0711 | 0.0703-0.0717 | traffic not known | traffic not known |
| 512x2048 | bfloat16 | triton_fused | 0.0226 | 0.0225-0.0231 | 185.8 | 77.8% |
| 512x2048 | bfloat16 | triton_unfused | 0.0301 | 0.0293-0.0307 | 278.7 | 116.7% |
| 512x2048 | bfloat16 | torch_eager | 0.3624 | 0.3615-0.3631 | traffic not known | traffic not known |
| 512x2048 | bfloat16 | torch_compile (see note) | 0.3152 | 0.3140-0.3154 | traffic not known | traffic not known |
| 1024x4096 | bfloat16 | triton_fused | 0.0779 | 0.0778-0.0786 | 215.6 | 90.2% |
| 1024x4096 | bfloat16 | triton_unfused | 0.1490 | 0.1478-0.1495 | 225.2 | 94.3% |
| 1024x4096 | bfloat16 | torch_eager | 1.5254 | 1.5237-1.5272 | traffic not known | traffic not known |
| 1024x4096 | bfloat16 | torch_compile (see note) | 1.3014 | 1.3001-1.3028 | traffic not known | traffic not known |

Marked rows:

- 1024x4096 bfloat16 torch_compile: Inductor skipped bf16 on this GPU; falls back to eager, not a fused baseline
- 128x512 bfloat16 torch_compile: Inductor skipped bf16 on this GPU; falls back to eager, not a fused baseline
- 256x1024 bfloat16 torch_compile: Inductor skipped bf16 on this GPU; falls back to eager, not a fused baseline
- 512x2048 bfloat16 torch_compile: Inductor skipped bf16 on this GPU; falls back to eager, not a fused baseline

## Predicted versus measured

L2 cache on this device: 4 MiB. The intermediate column is the size of the tensor that fusion avoids round-tripping; compare it against L2.

`L2-resident` is a judgement, not a measurement. The intermediate does not get the cache to itself -- it competes with the input and output streams flowing through it -- so an intermediate is only counted as resident below half of L2, and `marginal` covers the band from there up to the full capacity.

| shape | dtype | intermediate | L2-resident | predicted | measured | measured/predicted |
| --- | --- | ---: | :--: | ---: | ---: | ---: |
| 128x512 | float32 | 0.25 MiB | yes | 2.00x | 1.50x | 75% |
| 256x1024 | float32 | 1.00 MiB | yes | 2.00x | 1.22x | 61% |
| 512x2048 | float32 | 4.00 MiB | marginal | 2.00x | 1.84x | 92% |
| 1024x4096 | float32 | 16.00 MiB | no | 2.00x | 1.96x | 98% |
| 128x512 | bfloat16 | 0.12 MiB | yes | 2.00x | 1.33x | 66% |
| 256x1024 | bfloat16 | 0.50 MiB | yes | 2.00x | 1.31x | 66% |
| 512x2048 | bfloat16 | 2.00 MiB | yes | 2.00x | 1.33x | 67% |
| 1024x4096 | bfloat16 | 8.00 MiB | no | 2.00x | 1.91x | 96% |

## What these results do and do not establish

Established:

- The three kernels compute what the PyTorch references compute, across four
  shapes and two dtypes, within tolerances derived from dtype epsilon.
- Fusing RMSNorm into softmax does not cost accuracy. In bf16 it measurably
  improves it, because the intermediate is never rounded to storage precision.
- The fused kernel moves half as many bytes to and from HBM as the unfused
  pair, and roughly 7.5x fewer than naive eager PyTorch. This is arithmetic
  from shapes and dtype sizes, not a measurement.

Also established by this run, because it was made on hardware:

- The kernels are correct as *compiled* code, not merely as interpreted logic.
  The two can differ: the interpreter evaluates `tl.exp` with NumPy, while a
  GPU uses a faster approximate instruction.
- The predicted traffic reduction does convert into a proportional speedup,
  but only once the intermediate is too large to live in L2. See the
  predicted-versus-measured table above; the ratio climbs toward 100% as the
  working set grows past cache.

Still not established:

- That the traffic reduction is the *only* mechanism at work. Launch overhead
  dominates the smallest shapes, where a single fused launch replaces two
  regardless of bytes moved, and the two effects are not separated here.
- Anything about hardware other than the GPU this ran on. The L2 threshold
  that shapes these results is device-specific.
