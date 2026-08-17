"""RMSNorm followed by softmax, fused into one Triton kernel.

This is the point of the project. Running the two operations as separate
kernels forces the intermediate through global memory: the RMSNorm kernel
writes it out, and the softmax kernel immediately reads it back. Neither
transfer is needed, because the value was already on-chip.

Memory movement
---------------
Lives in global memory (HBM on a GPU):
    X  (M, N)  input,  read once
    W  (N,)    weight, read once per program instance
    Y  (M, N)  final softmax output, written once

Stays on-chip and is never written anywhere:
    H = rmsnorm(X, W), the (M, N) intermediate. In the unfused pipeline this
    costs M*N writes plus M*N reads. Here it is simply a live tile that flows
    from the normalization into the softmax. Note what "keeping it on-chip"
    means in Triton: there is no explicit scratchpad allocation, as there
    would be on an architecture with a programmer-managed SRAM. Fusion is
    achieved by *not* calling tl.store on the intermediate.

Tile boundaries:
    One program instance per row, grid = (M,). The whole row is resident, so
    all three reductions -- sum of squares, row max, sum of exponentials --
    read the same on-chip tile.

Traffic:
    M*N reads + M*N writes + N weight reads. Compare the unfused pair at
    2*M*N reads + 2*M*N writes: this halves it.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from kernels.common import as_2d, launch_config


@triton.jit
def _fused_rmsnorm_softmax_kernel(
    X,
    W,
    Y,
    stride_xm,
    stride_ym,
    N,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < N

    # Padding is 0.0 because the first reduction is a sum of squares.
    x = tl.load(X + row * stride_xm + cols, mask=mask, other=0.0).to(tl.float32)
    w = tl.load(W + cols, mask=mask, other=0.0).to(tl.float32)

    # ---- RMSNorm, in fp32 ----
    mean_square = tl.sum(x * x, axis=0) / N
    rstd = 1.0 / tl.sqrt(mean_square + eps)
    h = x * rstd * w

    # ---- softmax over the same resident tile ----
    # The padding value has to change here. It was 0.0 for the sum of squares,
    # but 0.0 would beat genuinely negative entries in the max reduction, so
    # the padding lanes are pushed to -inf before the max is taken.
    h = tl.where(mask, h, -float("inf"))
    h = h - tl.max(h, axis=0)
    numerator = tl.exp(h)
    denominator = tl.sum(numerator, axis=0)

    tl.store(Y + row * stride_ym + cols, numerator / denominator, mask=mask)


def fused_rmsnorm_softmax(
    x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6
) -> torch.Tensor:
    """softmax(rmsnorm(x, weight)) computed in a single pass over `x`."""
    x2d = as_2d(x)
    n_rows, n_cols = x2d.shape
    if weight.shape != (n_cols,):
        raise ValueError(f"weight must have shape ({n_cols},), got {tuple(weight.shape)}")

    out = torch.empty_like(x2d)
    block_size, num_warps = launch_config(n_cols, x2d.element_size())

    _fused_rmsnorm_softmax_kernel[(n_rows,)](
        x2d, weight, out,
        x2d.stride(0), out.stride(0),
        n_cols, eps,
        BLOCK_SIZE=block_size,
        num_warps=num_warps,
    )
    return out.reshape(x.shape)
