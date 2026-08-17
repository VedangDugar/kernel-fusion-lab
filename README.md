# Kernel Fusion Lab

Hand-written [Triton](https://triton-lang.org) kernels for RMSNorm and softmax, plus a fused
version that computes both in a single pass, with a harness that proves the fusion is correct
and quantifies exactly how much HBM traffic it eliminates.

RMSNorm and softmax are memory-bandwidth bound: they do very little arithmetic per byte they
touch. Run as two separate kernels, the intermediate tensor is written to HBM by the first and
immediately read back by the second, even though it was already sitting in on-chip memory.
Fusing them removes that round trip.

```
unfused                                   fused
-------                                   -----
HBM  --read X-->  [rmsnorm]               HBM  --read X-->  [rmsnorm]
                      |                                         |
HBM  <-write H---     |                                         |  H stays on-chip
HBM  --read H--->     |                                         v
                  [softmax]                                 [softmax]
HBM  <-write Y---                         HBM  <-write Y---

4*M*N elements moved                      2*M*N elements moved
```

## What this project establishes

**Correctness.** All three kernels match naive PyTorch references across four shapes
(128x512 through 1024x4096) and two dtypes (fp32, bf16). 32 of 32 asserted checks pass.
Tolerances are derived from each dtype's machine epsilon in `harness/config.py`, not tuned
until the tests went green.

**Traffic reduction, analytically.** The fused kernel moves **50% fewer bytes** than the
unfused pair and **7.5x fewer** than naive eager PyTorch. These are derived from shapes and
dtype sizes; the arithmetic is written out term by term in the docstring of
`harness/memory_model.py` so it can be checked by hand.

| shape | dtype | eager | unfused | fused | reduction |
| --- | --- | ---: | ---: | ---: | ---: |
| 1024x4096 | float32 | 240.05 MiB | 64.02 MiB | 32.02 MiB | 49.99% |
| 1024x4096 | bfloat16 | 120.02 MiB | 32.01 MiB | 16.01 MiB | 49.99% |

**Fusion improves accuracy in bf16.** This one was a surprise and is the most interesting
result in the repo — see below.

## What it does not establish

**No wall-clock speedup is claimed.** Every number above was produced on a CPU, using
Triton's interpreter (`TRITON_INTERPRET=1`), which executes kernel bodies sequentially in
Python on top of NumPy. It validates logic and nothing else: it models neither the memory
hierarchy nor occupancy nor latency. A timing taken there would describe the interpreter, not
the kernel, so the harness reports `not measured` in every timing field rather than
estimating one.

`notebooks/colab_benchmark.ipynb` collects real timings on a free Colab GPU. Until it is run,
the performance section of [`docs/RESULTS.md`](docs/RESULTS.md) stays empty by design.

## The bf16 finding

The first run of the correctness suite failed. The fused kernel and the unfused two-kernel
sequence disagreed by **8.3% relative error** in bf16, far more than output rounding could
account for.

The cause turned out to be the point of the project, viewed from the other side. The unfused
sequence *materialises* its intermediate `H = rmsnorm(X, W)` in memory, in the storage dtype.
`H` reaches a magnitude of about 12, where one bf16 ULP is 0.062. Softmax converts an
absolute perturbation of its logits into roughly the same relative perturbation of its
outputs, so writing `H` to memory as bf16 costs ~8% accuracy. The fused kernel never writes
`H`, so it never pays that cost.

Scored against an fp64 ground truth:

| pipeline | max relative error (bf16, 1024x4096) |
| --- | --- |
| fused kernel | 7.8e-3 |
| unfused sequence | 9.1e-2 |

The fused kernel is roughly **10x more accurate**, not merely different. So the test was
replaced with a stricter one rather than a looser one: `fusion_not_less_accurate` requires
the fused kernel to be at least as close to fp64 truth as the unfused sequence, with no
slack. The full investigation is reproducible via `scripts/bf16_precision_study.py`.

In fp32 the two pipelines agree to the bit, which is the expected result and a useful control.

## Running it

Triton ships Linux wheels only, so everything runs in a container. No GPU required.

```bash
docker build -t kfl:cpu .

# correctness suite
docker run --rm -e TRITON_INTERPRET=1 -e PYTHONPATH=/work \
    -v "$PWD:/work" -w /work kfl:cpu pytest -q

# full sweep, regenerates docs/RESULTS.md
docker run --rm -e TRITON_INTERPRET=1 -e PYTHONPATH=/work \
    -v "$PWD:/work" -w /work kfl:cpu python -m harness.sweep

# the bf16 precision investigation
docker run --rm -e TRITON_INTERPRET=1 -e PYTHONPATH=/work \
    -v "$PWD:/work" -w /work kfl:cpu python scripts/bf16_precision_study.py
```

`TRITON_INTERPRET` must be set before `triton` is imported, which is why it is passed as an
environment variable rather than set in Python.

For wall-clock numbers, open `notebooks/colab_benchmark.ipynb` in Google Colab, select a T4
runtime, and run all cells.

## Layout

```
kernels/          Triton kernels; each docstring states what lives in HBM,
                  what is staged on-chip, and where tile boundaries fall
  common.py         launch configuration and the row-residency guard
  rmsnorm.py
  softmax.py
  fused_rmsnorm_softmax.py
reference/        naive PyTorch implementations used as ground truth
harness/
  config.py         shapes, dtypes, and epsilon-derived tolerances
  correctness.py    asserted checks and reported measurements
  memory_model.py   analytical HBM traffic, derivations in the docstring
  benchmark.py      wall-clock timing; GPU only, never estimates
  sweep.py          runs everything, writes docs/RESULTS.md
tests/            pytest wrapper over the correctness harness
scripts/
  hello_kernel.py       toolchain smoke test
  bf16_precision_study.py
notebooks/        Colab GPU benchmark
docs/
  API_NOTES.md      Triton API details, read from the docs rather than recalled
  RESULTS.md        generated; do not edit
```

## Design notes

**One program instance per row, whole row resident.** The widest case is 4096 columns, which
is 16KB in fp32 — inside Triton's 64KB on-chip budget. So every case runs single-pass with no
tiling loop along N, and `kernels/common.py` raises rather than silently spilling if that
assumption is ever violated. This matters for the traffic argument: a kernel that tiles the
row would read the row more than once.

**Fusion in Triton means not calling `tl.store`.** Unlike architectures with a
programmer-managed scratchpad, Triton gives you no way to allocate on-chip memory. The
compiler owns shared memory and register allocation. An intermediate stays on-chip simply by
remaining a live tile that flows into the next computation.

**Masking is a correctness concern, not just a safety one.** Block sizes are padded to powers
of two, and the padding lanes participate in reductions. A sum of squares needs `other=0.0`;
a max reduction needs `other=-inf`. The fused kernel needs both, and switches between them
mid-kernel with a `tl.where` — using the wrong padding value produces silently wrong answers
rather than a crash.

**The weight vector is bracketed, not assumed.** `W` is read by every program instance.
Whether that costs `N` or `M*N` of HBM traffic depends on L2 behaviour, which the analytical
model cannot settle. Both bounds are reported; the reduction is 50% under ideal caching and
40% if `W` is re-fetched every row.

## Ground rules

These were fixed before any code was written and are visible in the results:

- No fabricated benchmarks. If something was not measured, the harness prints
  `not measured` rather than an estimate.
- Every kernel docstring explains its memory movement explicitly.
- Tolerances are derived from dtype epsilon. A failing kernel gets investigated and shown,
  never accommodated by widening a tolerance.
