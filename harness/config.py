"""Shapes, dtypes, and tolerances shared by every part of the harness.

Tolerances are derived from the machine epsilon of each dtype rather than
tuned until the tests pass. The rule for this project is that a failing
kernel gets shown, not accommodated, so these numbers are fixed here and
justified below.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import torch

# (rows, columns). The widest row is 4096 elements, which is 16KB in fp32 and
# 8KB in bf16 -- both comfortably inside the 64KB on-chip budget, so every
# case runs as a single-pass row-resident kernel with no tiling loop.
SHAPES: list[tuple[int, int]] = [
    (128, 512),
    (256, 1024),
    (512, 2048),
    (1024, 4096),
]

DTYPES: list[torch.dtype] = [torch.float32, torch.bfloat16]

EPS: float = 1e-6

# Machine epsilon, i.e. the gap between 1.0 and the next representable value.
#   fp32  has a 23-bit mantissa -> 2^-23 = 1.19e-7
#   bf16  has a  7-bit mantissa -> 2^-8  = 3.91e-3
MACHINE_EPS: dict[torch.dtype, float] = {
    torch.float32: 2.0 ** -23,
    torch.bfloat16: 2.0 ** -8,
}


@dataclass(frozen=True)
class Tolerance:
    rtol: float
    atol: float


# Both kernels and references accumulate in fp32 and round once on output, so
# the expected disagreement is a small number of units in the last place of
# the *output* dtype.
#
#   fp32:  rtol 1e-5 is ~84 ULP. Observed disagreement is ~4 ULP, so this
#          leaves roughly 20x of headroom without being loose enough to hide a
#          real logic error, which would be off by orders of magnitude.
#   bf16:  rtol 2e-2 is ~5 ULP. bf16 rounding alone accounts for 1 ULP on the
#          output, and the unfused pipeline rounds its intermediate too, which
#          softmax then amplifies. 5 ULP covers both.
#
# atol only matters where the reference is near zero. Softmax outputs at
# N=4096 sit around 2.4e-4, so atol=1e-6 stays well below the signal.
TOLERANCES: dict[torch.dtype, Tolerance] = {
    torch.float32: Tolerance(rtol=1e-5, atol=1e-6),
    torch.bfloat16: Tolerance(rtol=2e-2, atol=1e-6),
}

DTYPE_NAMES: dict[torch.dtype, str] = {
    torch.float32: "float32",
    torch.bfloat16: "bfloat16",
}

DTYPE_BYTES: dict[torch.dtype, int] = {
    torch.float32: 4,
    torch.bfloat16: 2,
}


def default_device() -> str:
    """Where kernels can actually execute in the current environment.

    Under the Triton interpreter, kernels run on the CPU via NumPy and the
    tensors must be CPU tensors. Compiled Triton requires device pointers and
    rejects CPU tensors outright, so on a real GPU the inputs have to live
    there. Getting this wrong surfaces as:

        ValueError: Pointer argument (at 0) cannot be accessed from Triton
    """
    if os.environ.get("TRITON_INTERPRET") == "1":
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def make_inputs(
    shape: tuple[int, int],
    dtype: torch.dtype,
    seed: int = 0,
    device: str | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Deterministic (x, weight) pair for a case, on the appropriate device.

    Values are always drawn on the CPU from a CPU generator and then moved, so
    the same seed yields bit-identical inputs whether the run is on CPU or GPU.
    Seeding a CUDA generator instead would produce a different stream and make
    CPU and GPU results incomparable.
    """
    device = device or default_device()
    generator = torch.Generator().manual_seed(seed)
    rows, cols = shape
    x = torch.randn(rows, cols, generator=generator, dtype=torch.float32)
    weight = torch.randn(cols, generator=generator, dtype=torch.float32)
    return (
        x.to(device=device, dtype=dtype),
        weight.to(device=device, dtype=dtype),
    )
