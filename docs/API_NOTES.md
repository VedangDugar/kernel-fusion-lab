# Triton API Notes

Everything here was read from the official Triton documentation or observed
directly in this repo's container. Nothing is recalled from memory. Items I
could not confirm are listed under [Open questions](#open-questions) rather
than guessed at.

Sources:

- Introduction / programming model — <https://triton-lang.org/main/programming-guide/chapter-1/introduction.html>
- Debugging and the interpreter — <https://triton-lang.org/main/programming-guide/chapter-3/debugging.html>
- Fused softmax tutorial — <https://triton-lang.org/main/getting-started/tutorials/02-fused-softmax.html>
- Layer normalization tutorial — <https://triton-lang.org/main/getting-started/tutorials/05-layer-norm.html>
- `do_bench` reference — <https://triton-lang.org/main/python-api/generated/triton.testing.do_bench.html>

## Verified environment

Observed inside the container built from this repo's `Dockerfile`:

| Component | Version |
| --- | --- |
| Python | 3.11 |
| Triton | 3.7.1 |
| PyTorch | 2.13.0+cpu |
| NumPy | 2.4.6 |
| pytest | 9.1.1 |

Triton ships Linux `x86_64` manylinux wheels. There is no supported Windows
wheel, which is why all development happens in the container.

## Programming model

Triton inverts CUDA's decomposition. CUDA is a *scalar program over blocked
threads*: you write code for one element and reason about how threads within a
block cooperate. Triton is a *blocked program over scalar threads*: you write
code that operates on whole tiles, and the compiler decides how to map that
tile onto threads.

The practical consequence is that intra-block concerns — coalescing, thread
swizzling, vectorization, shared memory allocation and synchronization,
async copy scheduling — are handled by the compiler, not by you. The
documentation lists these explicitly as automatic optimizations. What you
still own is the *tile shape* and the *memory access pattern*.

A kernel is a Python function decorated with `@triton.jit`, launched over a
grid with square-bracket syntax:

```python
kernel[grid](arg0, arg1, ..., BLOCK_SIZE=1024, num_warps=8)
```

`grid` is a tuple, or a callable taking the meta-parameter dict and returning
a tuple. Each entry in the grid produces one *program instance*, which
identifies itself with `tl.program_id(axis)`.

## Memory spaces

Three levels matter, and only one of them is named explicitly in the API:

| Level | In Triton | Managed by |
| --- | --- | --- |
| Global memory (DRAM / HBM on a GPU) | Raw pointers passed as kernel arguments | You, via `tl.load` / `tl.store` |
| On-chip SRAM (shared memory) | Never named directly | The compiler |
| Registers | Never named directly | The compiler |

This is the key difference from a hand-managed model. **You do not allocate
on-chip storage in Triton.** A value produced by `tl.load` is a tile that lives
in registers or shared memory as the compiler sees fit, and it stays resident
until it goes out of scope. Fusion in Triton therefore means *not calling
`tl.store` on an intermediate* — the intermediate simply remains a live tile
and is consumed by the next computation.

The size of what can stay resident is bounded by hardware. The layer-norm
tutorial encodes this as a hard limit:

```python
MAX_FUSED_SIZE = 65536 // x.element_size()   # 64KB per feature row
BLOCK_SIZE = min(MAX_FUSED_SIZE, triton.next_power_of_2(N))
if N > BLOCK_SIZE:
    raise RuntimeError("This layer norm doesn't support feature dim >= 64KB.")
```

So a row-resident kernel is only valid while one row fits in on-chip memory.
Beyond that you must either tile the row and loop (the layer-norm approach,
which reads the row more than once) or accept spilling.

## Tile constraints

- **Block sizes must be powers of two.** Stated directly in the softmax
  tutorial: "each block must have a power-of-two number of elements". Use
  `triton.next_power_of_2(n)` to round up, then mask off the tail.
- Tile shapes must be compile-time constants, declared as `tl.constexpr`
  parameters. They participate in the JIT cache key, so each distinct value
  triggers a separate compilation.
- `triton.cdiv(a, b)` is ceiling division, used for grid sizing.

## Masking

Because block sizes are padded to powers of two, nearly every load and store
needs a mask to stay in bounds:

```python
mask = offsets < n_elements
x = tl.load(ptr + offsets, mask=mask, other=0.0)
tl.store(out_ptr + offsets, val, mask=mask)
```

The `other=` value matters for correctness of reductions, not just safety. The
softmax tutorial loads with `other=-float('inf')` precisely so that padding
lanes cannot win a `tl.max`. A sum reduction wants `other=0.0` instead. Getting
this wrong produces wrong answers rather than crashes.

## Language surface used in this project

Confirmed in the tutorials above:

| Call | Purpose |
| --- | --- |
| `tl.program_id(axis)` | Index of this program instance |
| `tl.num_programs(axis)` | Total instances along an axis |
| `tl.arange(start, end)` | Tile of consecutive offsets; bounds must be constexpr |
| `tl.load(ptr, mask=, other=)` | Global memory into a tile |
| `tl.store(ptr, val, mask=)` | Tile back to global memory |
| `tl.sum(x, axis=)` / `tl.max(x, axis=)` | Reductions over a tile |
| `tl.exp(x)` / `tl.sqrt(x)` | Elementwise math |
| `tl.where(cond, a, b)` | Elementwise select |
| `tl.zeros(shape, dtype=)` | Zero-initialized accumulator tile |
| `x.to(tl.float32)` | Cast a tile |
| `tl.range(start, stop, step, num_stages=)` | Loop with software pipelining |

Note from the tutorial source: `tl.exp` is fast but approximate, comparable to
CUDA's `__expf`. That is relevant when choosing correctness tolerances — some
of the error against PyTorch is expected and is not a bug in the kernel.

## Launch-time tuning parameters

Passed as keyword arguments at launch, not declared in the signature:

- `num_warps` — how many warps cooperate on one program instance. The layer-norm
  tutorial derives it as `min(max(BLOCK_SIZE // 256, 1), 8)`; the softmax
  tutorial hardcodes `8`.
- `num_stages` — software pipelining depth.
- `num_ctas` — cooperative thread array count.

## CPU execution: the interpreter

Setting `TRITON_INTERPRET=1` makes all Triton kernels bypass compilation and
run on the CPU, simulated with NumPy equivalents of each `tl.*` operation.
Program instances are executed sequentially.

**The variable must be set before `triton` is imported.** Setting it later in
the process has no effect.

```bash
TRITON_INTERPRET=1 python scripts/hello_kernel.py
```

Confirmed working in this repo: `scripts/hello_kernel.py` runs a masked vector
add through the interpreter and matches `torch` bitwise.

What the interpreter is good for: validating kernel logic, masking, and
reduction correctness with no GPU. `print` and `pdb`/`breakpoint()` work
inside kernel bodies.

What it is **not** good for: anything about performance. It does not model
memory hierarchy, occupancy, or latency, and it runs orders of magnitude
slower than hardware. No timing number produced under the interpreter means
anything, so this project does not report any.

## Benchmarking

Verified signature:

```python
triton.testing.do_bench(fn, warmup=25, rep=100, grad_to_none=None,
                        quantiles=None, return_mode='mean')
```

- `warmup` and `rep` are durations in **milliseconds**, not iteration counts.
- `quantiles=[0.5, 0.2, 0.8]` returns median, 20th, and 80th percentile in ms
  and causes `return_mode` to be ignored. The tutorials use exactly this triple
  to report a median with error bars, which is the right call on a shared or
  thermally throttled GPU.
- `grad_to_none=[x]` prevents gradient accumulation from polluting backward-pass
  timings.

`triton.testing.perf_report` plus `triton.testing.Benchmark` generate sweep
tables and plots; the tutorials use them to sweep one axis against several
providers.

## Baseline expectations

Worth recording before measuring anything, so the results are not read
optimistically after the fact. The softmax tutorial's own published numbers
show the handwritten Triton kernel roughly **at parity** with `torch.softmax`
across most of the sweep — sometimes ahead, sometimes behind — while beating a
naive unfused PyTorch composition by roughly 4x.

The large win is therefore against *unfused eager* code, not against PyTorch's
own fused kernels. `torch.compile` fuses these operations and emits Triton
itself, so it is a genuinely hard baseline. This project reports whatever the
measurement shows against both.

## Open questions

Deferred until they actually block something, rather than guessed at now:

1. Whether `triton.testing.do_bench` runs at all under `TRITON_INTERPRET=1`. It
   uses CUDA events on GPU; the CPU path is unverified. Benchmarking is
   GPU-only in this project regardless, so this only affects whether the
   harness needs to guard the import.
2. Which `tl.*` operations, if any, are unsupported or subtly different in the
   interpreter. Community reports mention occasional gaps. Any divergence found
   will be recorded here.
3. Exact shared-memory capacity of the benchmark GPU, needed to justify tile
   sizes. Obtainable at runtime from
   `driver.active.utils.get_device_properties(idx)["max_shared_mem"]`, so it
   gets filled in during the benchmark run rather than assumed.
