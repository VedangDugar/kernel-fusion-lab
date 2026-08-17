"""Launch configuration shared by every kernel in this package.

All three kernels use the same decomposition: one program instance per row,
with the entire row held on-chip as a single tile. That is only valid while a
row fits in on-chip memory, which `check_row_fits` enforces.
"""

from __future__ import annotations

import torch
import triton

# Triton's practical ceiling for a resident row, taken from the layer-norm
# tutorial's MAX_FUSED_SIZE guard. Beyond this a kernel must tile the row and
# loop over it, which means reading the row more than once and changes the
# memory-traffic argument this project is built on.
MAX_RESIDENT_BYTES = 65536


def check_row_fits(n_cols: int, element_size: int) -> None:
    """Fail loudly rather than silently spilling."""
    row_bytes = n_cols * element_size
    if row_bytes > MAX_RESIDENT_BYTES:
        raise ValueError(
            f"row of {n_cols} x {element_size}B = {row_bytes}B exceeds the "
            f"{MAX_RESIDENT_BYTES}B on-chip budget; this kernel family assumes "
            "a row-resident tile and does not implement a tiling loop"
        )


def launch_config(n_cols: int, element_size: int) -> tuple[int, int]:
    """Return (BLOCK_SIZE, num_warps) for a row of `n_cols` elements.

    BLOCK_SIZE is rounded up to a power of two because Triton requires it;
    the tail lanes are handled by masking at every load and store.

    The num_warps heuristic is the one used by the official layer-norm
    tutorial: roughly one warp per 256 elements, clamped to [1, 8].
    """
    check_row_fits(n_cols, element_size)
    block_size = triton.next_power_of_2(n_cols)
    num_warps = min(max(block_size // 256, 1), 8)
    return block_size, num_warps


def as_2d(x: torch.Tensor) -> torch.Tensor:
    """Collapse leading dimensions so kernels only deal with (M, N)."""
    return x.reshape(-1, x.shape[-1])
