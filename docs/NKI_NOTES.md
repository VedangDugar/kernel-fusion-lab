# NKI API Notes

Written while porting the fused kernel to AWS Trainium's Neuron Kernel
Interface. Everything below was verified by introspecting the installed
package or by running code against the CPU simulator. Several details
contradict what a language model would plausibly guess, and those are called
out, because guessing here produces code that looks right and does not exist.

## Verified environment

Installed by `Dockerfile.nki` from `pip.repos.neuron.amazonaws.com`:

| Component | Version |
| --- | --- |
| nki | 0.5.0+28631259367.ga768afa6 |
| neuronx-cc | 2.26.6360.0+6f180f47 |
| Python | 3.11 |
| numpy | 1.26.4 (pinned down by neuronx-cc from 2.4.6) |

Linux x86_64 only. Note that `neuronx-cc` downgrades NumPy below 2.x, which is
why the Neuron toolchain lives in a separate image from the Triton one.

## Corrections to plausible-looking guesses

These are the specific places where an invented API would have compiled in
someone's head and failed in reality:

| Guess | Reality |
| --- | --- |
| `nki.simulate_kernel(f)(...)` | The entry point is **`nki.simulate(f)(...)`**. There is no `simulate_kernel` attribute on the `nki` module. |
| `nl.arange(0, BLOCK)` | **Does not exist.** Indexing uses `nl.ds(start, size)` for a dynamic slice, or ordinary Python slicing. |
| `a + b` on tiles | **`NkiTensor` does not overload arithmetic operators.** `TypeError: unsupported operand type(s) for +`. Every operation is an explicit call: `nl.add`, `nl.subtract`, `nl.multiply`, `nl.divide`, `nl.square`. |
| `nl.affine_range(n)` for loops | Exists but is **deprecated** and slated for removal; the docstring directs you to `nl.static_range`. |
| A weight vector can be passed as shape `(N,)` | Every tile has a partition axis. A 1-D length-N tensor asks for N partitions and exceeds the 128 limit. Pass it as `(1, N)`. |

## Entry points

```python
@nki.jit
def my_kernel(a, b): ...

result = nki.simulate(my_kernel)(a_np, b_np)   # CPU simulator
```

`nki.jit` inspects its arguments to pick a framework: `torch.Tensor` uses the
PyTorch integration, `jax.Array` uses JAX, and `np.ndarray` compiles and runs
standalone. `nki.simulate` returns results in whichever format it was given.

Target hardware is selected with the `NEURON_PLATFORM_TARGET_OVERRIDE`
environment variable, accepting `trn1|inf2|gen2`, `trn2|gen3`, or `trn3|gen4`.
The Logical NeuronCore degree is set at the call site with bracket syntax,
`kernel[2](args)`, defaulting to 1.

`nki.simulate` is marked `<<experimental>>` in its own docstring, as are
`nl.load`, `nl.store`, `nl.sum`, `nl.max`, and `nl.exp`. Treat the surface as
unstable across releases.

## Memory spaces

Unlike Triton, NKI names the memory hierarchy directly and you place data in
it yourself:

| Space | Constant | Role |
| --- | --- | --- |
| Device DRAM | `nl.hbm`, `nl.private_hbm`, `nl.shared_hbm` | Where inputs and outputs live |
| On-chip SRAM | `nl.sbuf` | Working set; the default `buffer=` for `nl.ndarray` |
| Accumulation buffer | `nl.psum` | Tensor-engine matmul output |

```python
out = nl.ndarray(shape, dtype, buffer=nl.shared_hbm)  # declare an HBM output
tile = nl.load(src, dtype=nl.float32)                 # HBM -> SBUF
nl.store(dst, value)                                  # SBUF -> HBM
```

Predicates `nl.is_hbm`, `nl.is_sbuf`, `nl.is_psum`, and `nl.is_on_chip` are
available for asserting where a tensor actually lives.

PSUM is only relevant to matrix multiplication. A pure reduction kernel like
this one runs on the vector engine against SBUF and never touches PSUM.

## The constraint that reshapes the kernel

SBUF is physically 128 partitions wide, and NKI exposes that as a hard limit:

```python
nl.tile_size.pmax  # 128
```

Two consequences drive the whole port:

1. **A tile's first axis is the partition axis and cannot exceed 128.** Any
   matrix with more than 128 rows must be processed in blocks.
2. **Reductions may only run along free axes.** From the `nl.sum` docstring:
   *"must be free dimensions, not partition dimension (0); can only be the
   last contiguous dim(s) of the tile."*

Together these force the layout. A row-wise reduction requires rows on the
partition axis and columns on the free axis, so the kernel becomes a loop over
`ceil(M / 128)` blocks of shape `(128, N)`, reducing along `axis=1`.

This is the sharpest contrast with Triton, where one row per program instance
is natural and the compiler hides the partitioning entirely.

Other published tile limits:

| Constant | Value |
| --- | --- |
| `pmax` | 128 |
| `gemm_stationary_fmax` | 128 |
| `gemm_moving_fmax` | 512 |
| `psum_fmax` | 512 |
| `psum_bank_fmax` | 512 |
| `psum_bank_fmax_bytes` | 2048 |
| `bn_stats_fmax` | 512 |

Some `nl.tile_size` attributes (`psum_num_banks`, `sbuf_fmax_bytes`) are
computed lazily from the active backend and raise
`RuntimeError: No backend set` when read outside a kernel context.

## Broadcasting

Verified working in the fused kernel:

- A `(128, 1)` tile broadcasts along the free axis against `(128, N)`. This is
  how the per-row reciprocal norm is applied.
- A `(1, N)` tile broadcasts along the partition axis against `(128, N)`. This
  is how the shared weight vector is applied.

## Masking, or the absence of it

Triton pads every block to a power of two, so padding lanes participate in
reductions and each load needs a correct `other=` value. NKI tiles are exactly
the shape you ask for, so this class of bug does not arise. The trade is that
NKI gives you no help when the row count is not a multiple of 128; that has to
be handled explicitly.

## Fusion means something different in each language

- **Triton:** the compiler owns on-chip storage. An intermediate stays
  on-chip by virtue of you not calling `tl.store` on it. There is no way to
  ask where it physically lives.
- **NKI:** SBUF is named and you place tiles in it yourself. An intermediate
  stays on-chip because you never hand it to `nl.store`, and you can assert
  where it is with `nl.is_sbuf`.

The end result is the same traffic reduction. The difference is how much the
compiler decides for you.

## What the simulator does and does not establish

`nki.simulate` executes the kernel on the CPU and returns numerically
meaningful results. It says nothing about performance, occupancy, DMA
scheduling, or instruction selection on real silicon. No Trainium hardware was
used anywhere in this project and no timing is claimed for the NKI path.

## Verified results

`scripts/nki_verify.py`, fp32, against a float64 NumPy ground truth:

| shape | max abs err | max rel err |
| --- | ---: | ---: |
| 128x512 | 1.563e-07 | 1.277e-06 |
| 256x1024 | 3.242e-07 | 1.604e-06 |
| 512x2048 | 2.858e-07 | 2.077e-06 |
| 1024x4096 | 3.531e-07 | 2.829e-06 |

Comparable to the Triton kernel's fp32 error on the same shapes, which is the
expected outcome: both accumulate in fp32 over the same number of terms.
