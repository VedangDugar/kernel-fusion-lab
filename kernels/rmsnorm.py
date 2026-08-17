"""RMSNorm as a single Triton kernel.

Memory movement
---------------
Lives in global memory (HBM on a GPU):
    X  (M, N)  input,  read once
    W  (N,)    weight, read once per program instance, served from L2 in practice
    Y  (M, N)  output, written once

Staged on-chip:
    One full row of X, loaded as a single BLOCK_SIZE tile. Triton places this
    in registers or shared memory; the language does not let you choose, and
    the compiler owns that decision. The row stays resident for the whole
    kernel body, so the sum of squares, the reciprocal norm, and the scaled
    output are all computed without touching global memory again.

Tile boundaries:
    One program instance per row, grid = (M,). BLOCK_SIZE is the row width
    rounded up to a power of two, so the final lanes of the tile are padding
    and every load and store is masked. There is no tiling loop along N; the
    row is assumed to fit on-chip and `check_row_fits` enforces that.

Traffic:
    M*N reads + M*N writes + N weight reads, i.e. one read and one write per
    element. That is the floor for this operation as a standalone kernel.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from kernels.common import as_2d, launch_config


@triton.jit
def _rmsnorm_kernel(
    X,          # *dtype, (M, N) input
    W,          # *dtype, (N,)   per-column scale
    Y,          # *dtype, (M, N) output
    stride_xm,  # elements to advance one row of X
    stride_ym,  # elements to advance one row of Y
    N,          # true row width, may be < BLOCK_SIZE
    eps,        # added under the sqrt for numerical stability
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < N

    # other=0.0 is the correct padding value here: padding lanes contribute
    # nothing to a sum of squares. A max reduction would need -inf instead.
    x = tl.load(X + row * stride_xm + cols, mask=mask, other=0.0).to(tl.float32)
    w = tl.load(W + cols, mask=mask, other=0.0).to(tl.float32)

    # Reduce in fp32 regardless of input dtype. Accumulating N terms in bf16
    # would lose most of the mantissa well before N reaches 4096.
    mean_square = tl.sum(x * x, axis=0) / N
    rstd = 1.0 / tl.sqrt(mean_square + eps)

    y = x * rstd * w
    # tl.store casts the fp32 tile down to Y's element type implicitly.
    tl.store(Y + row * stride_ym + cols, y, mask=mask)


def rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Row-wise RMSNorm. Returns a tensor with the same shape and dtype as `x`."""
    x2d = as_2d(x)
    n_rows, n_cols = x2d.shape
    if weight.shape != (n_cols,):
        raise ValueError(f"weight must have shape ({n_cols},), got {tuple(weight.shape)}")

    out = torch.empty_like(x2d)
    block_size, num_warps = launch_config(n_cols, x2d.element_size())

    _rmsnorm_kernel[(n_rows,)](
        x2d, weight, out,
        x2d.stride(0), out.stride(0),
        n_cols, eps,
        BLOCK_SIZE=block_size,
        num_warps=num_warps,
    )
    return out.reshape(x.shape)
