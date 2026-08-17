"""Row-wise softmax as a single Triton kernel.

Memory movement
---------------
Lives in global memory (HBM on a GPU):
    X  (M, N)  input,  read once
    Y  (M, N)  output, written once

Staged on-chip:
    One full row of X as a single BLOCK_SIZE tile. Both reductions the
    algorithm needs -- the row maximum and the sum of exponentials -- run
    against that resident tile, so the row is never re-read from global
    memory. A naive eager implementation re-reads it roughly five times.

Tile boundaries:
    One program instance per row, grid = (M,). BLOCK_SIZE is the row width
    rounded to a power of two; tail lanes are padding and every access is
    masked.

Traffic:
    M*N reads + M*N writes.

Numerical note:
    The max is subtracted before exponentiating so that exp never sees a
    large positive argument. Softmax is invariant to that shift.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from kernels.common import as_2d, launch_config


@triton.jit
def _softmax_kernel(
    X,          # *dtype, (M, N) input
    Y,          # *dtype, (M, N) output
    stride_xm,
    stride_ym,
    N,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < N

    # other=-inf, not 0.0: padding lanes must lose the max reduction. With 0.0
    # a row of all-negative values would take its maximum from the padding and
    # silently produce the wrong answer.
    x = tl.load(X + row * stride_xm + cols, mask=mask, other=-float("inf")).to(tl.float32)

    x = x - tl.max(x, axis=0)
    # exp(-inf) == 0, so padding lanes drop out of the sum on their own.
    numerator = tl.exp(x)
    denominator = tl.sum(numerator, axis=0)

    tl.store(Y + row * stride_ym + cols, numerator / denominator, mask=mask)


def softmax(x: torch.Tensor) -> torch.Tensor:
    """Softmax over the last dimension. Same shape and dtype as `x`."""
    x2d = as_2d(x)
    n_rows, n_cols = x2d.shape

    out = torch.empty_like(x2d)
    block_size, num_warps = launch_config(n_cols, x2d.element_size())

    _softmax_kernel[(n_rows,)](
        x2d, out,
        x2d.stride(0), out.stride(0),
        n_cols,
        BLOCK_SIZE=block_size,
        num_warps=num_warps,
    )
    return out.reshape(x.shape)
