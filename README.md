# Kernel Fusion Lab

[![Open the GPU benchmark in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/VedangDugar/kernel-fusion-lab/blob/main/notebooks/colab_benchmark.ipynb)

Hand-written [Triton](https://triton-lang.org) kernels for RMSNorm and softmax, plus a fused
version that computes both in a single pass, with a harness that proves the fusion is correct
and quantifies exactly how much HBM traffic it eliminates. The fused kernel is also ported to
[NKI](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/nki/index.html) for AWS
Trainium and validated on Neuron's CPU simulator.

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

## The Trainium port

The fused kernel is also implemented in [NKI](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/nki/index.html),
the Neuron Kernel Interface for AWS Trainium, and validated on Neuron's CPU simulator
(`kernels_nki/`, verified by `scripts/nki_verify.py`). No Trainium hardware was involved and
no timing is claimed for it — the simulator establishes numerics only.

| shape | max abs err | max rel err |
| --- | ---: | ---: |
| 128x512 | 1.563e-07 | 1.277e-06 |
| 1024x4096 | 3.531e-07 | 2.829e-06 |

The interesting part is that the *decomposition had to change*. A NeuronCore's SBUF is
physically 128 partitions wide, and NKI exposes that as a hard limit: a tile's first axis is
the partition axis and cannot exceed `nl.tile_size.pmax` = 128, and **reductions may only run
along free axes**, never the partition axis. A row-wise reduction therefore requires rows on
partitions and columns on the free axis, so the kernel becomes an explicit loop over
`ceil(M / 128)` blocks of shape `(128, N)`. Triton's one-row-per-program-instance structure
does not carry over at all.

The two languages also mean different things by fusion. Triton's compiler owns on-chip
storage, so an intermediate stays resident by virtue of you *not* calling `tl.store`. NKI
names SBUF explicitly and you place tiles in it yourself. Same traffic reduction, different
amount of control.

[`docs/NKI_NOTES.md`](docs/NKI_NOTES.md) records the verified API, including several places
where the obvious guess is wrong — the simulator entry point is `nki.simulate`, not
`nki.simulate_kernel`; `nl.arange` does not exist; and `NkiTensor` does not overload
arithmetic operators, so every operation is an explicit `nl.add` / `nl.multiply` call.

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

The Trainium port needs a second image, because `neuronx-cc` pins NumPy below 2.x and would
otherwise conflict with the Triton environment:

```bash
docker build -f Dockerfile.nki -t kfl:nki .

docker run --rm -e PYTHONPATH=/work -v "$PWD:/work" -w /work kfl:nki \
    python scripts/nki_verify.py
```

`TRITON_INTERPRET` must be set before `triton` is imported, which is why it is passed as an
environment variable rather than set in Python.

For wall-clock numbers, open the notebook in Google Colab via the badge at the top of this
file, or directly:

<https://colab.research.google.com/github/VedangDugar/kernel-fusion-lab/blob/main/notebooks/colab_benchmark.ipynb>

Set `Runtime -> Change runtime type -> T4 GPU`, then `Runtime -> Run all`. The first cell
fails fast if no GPU is attached, so a misconfigured runtime cannot silently produce
meaningless numbers.

## Layout

```
kernels/          Triton kernels; each docstring states what lives in HBM,
                  what is staged on-chip, and where tile boundaries fall
  common.py         launch configuration and the row-residency guard
  rmsnorm.py
  softmax.py
  fused_rmsnorm_softmax.py
kernels_nki/      the same fused kernel for AWS Trainium
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
  hello_kernel.py       Triton toolchain smoke test
  bf16_precision_study.py
  nki_hello.py          Neuron toolchain smoke test
  nki_verify.py         NKI kernel against a float64 ground truth
notebooks/        Colab GPU benchmark
docs/
  API_NOTES.md      Triton API details, read from the docs rather than recalled
  NKI_NOTES.md      NKI API details, verified against the installed package
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
