"""The fused RMSNorm -> softmax kernel, ported to NKI for AWS Trainium.

This is the same computation as `kernels/fused_rmsnorm_softmax.py`, but the
decomposition is different, and the reason it is different says something
about the two architectures.

Triton versus NKI
-----------------
In Triton the natural unit is one row per program instance, because the
compiler owns on-chip storage and a reduction can run over any axis of a tile.
That does not transfer. A NeuronCore's SBUF is physically organised as 128
partitions, and NKI exposes that directly:

  * A tile's first axis is the *partition* axis and may not exceed
    `nl.tile_size.pmax`, which is 128.
  * Reductions may only run along *free* axes. `nl.sum(x, axis=0)` is not
    expressible, because axis 0 is the partition axis.

So rows have to be placed on the partition axis and columns on the free axis,
and the kernel walks the matrix in blocks of 128 rows. Where the Triton
version has an implicit grid of M program instances, this has an explicit
loop over ceil(M / 128) blocks.

The second difference is that on-chip memory is *named* here. `nl.load` moves
HBM to SBUF and `nl.store` moves SBUF back. In Triton there is no such
vocabulary; the compiler decides. Fusion in Triton means declining to call
`tl.store`. Fusion here means the SBUF tile is never handed back to `nl.store`
until the final result exists.

Memory movement
---------------
HBM:
    x    (M, N)   input,  read once
    w    (1, N)   weight, read once per row block
    out  (M, N)   output, written once

SBUF:
    One (128, N) tile of x, resident for an entire block. Every intermediate
    -- the squares, the normalised values, the shifted logits, the
    exponentials -- lives in SBUF and is never stored to HBM.

PSUM is not used. It is the accumulation buffer for the tensor engine, and
this kernel performs no matrix multiplication; its reductions run on the
vector engine against SBUF directly.

Tile boundaries:
    Partition axis: 128 rows, the hardware maximum.
    Free axis:      the full row of N columns. At N = 4096 in fp32 that is
                    16KB per partition, which fits SBUF comfortably.
"""

from __future__ import annotations

import math

import nki
import nki.language as nl


@nki.jit
def fused_rmsnorm_softmax(x, w, eps=1e-6):
    """softmax(rmsnorm(x, w)) in a single pass over x.

    Args:
        x: (M, N) tensor in HBM.
        w: (1, N) tensor in HBM. Shaped 2D rather than 1D because every NKI
           tile has a partition axis; a length-N vector would otherwise be
           asked to occupy N partitions and blow past the 128 limit.
        eps: added under the reciprocal square root.

    Returns:
        (M, N) tensor in HBM with the same dtype as x.
    """
    n_rows, n_cols = x.shape
    out = nl.ndarray((n_rows, n_cols), dtype=x.dtype, buffer=nl.shared_hbm)

    rows_per_block = nl.tile_size.pmax
    n_blocks = math.ceil(n_rows / rows_per_block)

    # Loaded once outside the loop: the weight is the same for every block.
    weight = nl.load(w, dtype=nl.float32)

    for block in nl.static_range(n_blocks):
        rows = nl.ds(block * rows_per_block, rows_per_block)

        # HBM -> SBUF. Promoted to fp32 on load, matching the Triton kernel:
        # accumulating 4096 bf16 terms would destroy the mantissa.
        tile = nl.load(x[rows, :], dtype=nl.float32)

        # ---- RMSNorm ----
        # axis=1 is the free axis. A reduction over axis=0 would be the
        # partition axis and is not expressible in NKI.
        mean_square = nl.divide(nl.sum(nl.square(tile), axis=1, keepdims=True), n_cols)
        rstd = nl.rsqrt(nl.add(mean_square, eps))

        # rstd is (128, 1) and broadcasts along the free axis; weight is
        # (1, N) and broadcasts along the partition axis.
        normalized = nl.multiply(tile, rstd)
        hidden = nl.multiply(normalized, weight)

        # ---- softmax, on the same resident tile ----
        # No masking concern here, unlike the Triton version: NKI tiles are
        # exactly (128, N) with no power-of-two padding, so there are no
        # padding lanes that could win the max reduction.
        shifted = nl.subtract(hidden, nl.max(hidden, axis=1, keepdims=True))
        exponentiated = nl.exp(shifted)
        denominator = nl.sum(exponentiated, axis=1, keepdims=True)

        # The only SBUF -> HBM transfer in the kernel.
        nl.store(out[rows, :], nl.divide(exponentiated, denominator))

    return out
