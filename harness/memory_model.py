"""Analytical HBM traffic for the fused and unfused pipelines.

Nothing here is measured. Every number is derived from the shape and the
dtype size, and every derivation is written out below so it can be checked
with a pencil. This is a *prediction*; the benchmark harness separately
measures whether reality agrees.

Notation
--------
    M   rows
    N   columns
    S   bytes per element (4 for fp32, 2 for bf16)

Counts below are in elements. Bytes are elements * S.

Caching assumption
------------------
The weight vector W has N elements and is read by every one of the M program
instances. Whether that costs N or M*N of HBM traffic depends entirely on
whether it stays resident in L2, which is a hardware behaviour this model
cannot settle by itself. Both bounds are therefore reported:

    ideal  -- W is fetched from HBM once and served from L2 thereafter
    cold   -- W is re-fetched from HBM by every row

Reality sits at the ideal end for these shapes, because W is at most 16KB
against an L2 measured in megabytes, but the cold bound is shown so the
claimed reduction is bracketed rather than asserted.

Derivations
-----------
FUSED, one kernel, one pass over X:

    read  X                     M*N
    read  W                     N        (ideal)   or  M*N  (cold)
    write Y                     M*N
    ------------------------------------------
    fused_ideal  =  2*M*N + N
    fused_cold   =  3*M*N

UNFUSED, two Triton kernels. The intermediate H is a real tensor in memory:

    kernel 1, rmsnorm
        read  X                 M*N
        read  W                 N        (ideal)   or  M*N  (cold)
        write H                 M*N
    kernel 2, softmax
        read  H                 M*N
        write Y                 M*N
    ------------------------------------------
    unfused_ideal =  4*M*N + N
    unfused_cold  =  5*M*N

The difference is exactly 2*M*N: one write of H and one read of H. That is
the traffic fusion removes, and it is why the reduction tends to 50% as M
grows and the N weight term becomes negligible.

EAGER PyTorch, for reference. Every intermediate is materialised, so this is
the baseline a fused kernel is actually competing against in real code.

    rmsnorm
        sq  = x * x             read  M*N            write M*N
        ms  = sq.mean(-1)       read  M*N            write M
        r   = rsqrt(ms + eps)   read  M              write M
        xn  = x * r             read  M*N + M        write M*N
        h   = xn * w            read  M*N + N        write M*N
        reads  = 4*M*N + 2*M + N
        writes = 3*M*N + 2*M
        subtotal = 7*M*N + 4*M + N

    softmax
        mx  = x.max(-1)         read  M*N            write M
        z   = x - mx            read  M*N + M        write M*N
        num = exp(z)            read  M*N            write M*N
        den = num.sum(-1)       read  M*N            write M
        y   = num / den         read  M*N + M        write M*N
        reads  = 5*M*N + 2*M
        writes = 3*M*N + 2*M
        subtotal = 8*M*N + 4*M

    eager = 15*M*N + 8*M + N

So against eager the fused kernel moves roughly 15/2 = 7.5x less data, and
against a competently written pair of unfused kernels it moves 2x less.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from harness.config import DTYPE_BYTES, DTYPE_NAMES


@dataclass(frozen=True)
class TrafficModel:
    shape: tuple[int, int]
    dtype: str
    element_bytes: int
    eager_bytes: int
    unfused_bytes: int
    fused_bytes: int
    unfused_bytes_cold: int
    fused_bytes_cold: int

    @property
    def reduction_pct(self) -> float:
        """Fused versus unfused, ideal-caching bound."""
        return 100.0 * (self.unfused_bytes - self.fused_bytes) / self.unfused_bytes

    @property
    def reduction_pct_cold(self) -> float:
        return 100.0 * (self.unfused_bytes_cold - self.fused_bytes_cold) / self.unfused_bytes_cold

    @property
    def speedup_vs_eager(self) -> float:
        """Traffic ratio, not a time ratio. Names the bound, not the outcome."""
        return self.eager_bytes / self.fused_bytes


def model(shape: tuple[int, int], dtype: torch.dtype) -> TrafficModel:
    rows, cols = shape
    m, n = rows, cols
    s = DTYPE_BYTES[dtype]

    mn = m * n

    fused_ideal = 2 * mn + n
    fused_cold = 3 * mn

    unfused_ideal = 4 * mn + n
    unfused_cold = 5 * mn

    eager = 15 * mn + 8 * m + n

    return TrafficModel(
        shape=shape,
        dtype=DTYPE_NAMES[dtype],
        element_bytes=s,
        eager_bytes=eager * s,
        unfused_bytes=unfused_ideal * s,
        fused_bytes=fused_ideal * s,
        unfused_bytes_cold=unfused_cold * s,
        fused_bytes_cold=fused_cold * s,
    )


def model_all(shapes, dtypes) -> list[TrafficModel]:
    return [model(shape, dtype) for dtype in dtypes for shape in shapes]


def _mib(n_bytes: int) -> str:
    return f"{n_bytes / (1024 * 1024):.2f}"


def format_table(models: list[TrafficModel]) -> str:
    header = (
        f"{'shape':>12} {'dtype':>9} {'eager MiB':>10} {'unfused MiB':>12} "
        f"{'fused MiB':>10} {'reduction':>10} {'cold bound':>11} {'vs eager':>9}"
    )
    lines = [header, "-" * len(header)]
    for m in models:
        lines.append(
            f"{str(m.shape):>12} {m.dtype:>9} {_mib(m.eager_bytes):>10} "
            f"{_mib(m.unfused_bytes):>12} {_mib(m.fused_bytes):>10} "
            f"{m.reduction_pct:>9.2f}% {m.reduction_pct_cold:>10.2f}% "
            f"{m.speedup_vs_eager:>8.2f}x"
        )
    return "\n".join(lines)


def format_markdown(models: list[TrafficModel]) -> str:
    lines = [
        "| shape | dtype | eager (MiB) | unfused (MiB) | fused (MiB) | reduction | reduction (cold W) | traffic ratio vs eager |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for m in models:
        lines.append(
            f"| {m.shape[0]}x{m.shape[1]} | {m.dtype} | {_mib(m.eager_bytes)} | "
            f"{_mib(m.unfused_bytes)} | {_mib(m.fused_bytes)} | "
            f"{m.reduction_pct:.2f}% | {m.reduction_pct_cold:.2f}% | "
            f"{m.speedup_vs_eager:.2f}x |"
        )
    return "\n".join(lines)


if __name__ == "__main__":
    from harness.config import DTYPES, SHAPES

    print(format_table(model_all(SHAPES, DTYPES)))
