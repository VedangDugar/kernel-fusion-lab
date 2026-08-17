"""Smoke test: a trivial Triton kernel executed end to end on the CPU.

This exists to prove the toolchain works before any real kernel is written.
It adds two vectors and checks the result against PyTorch.

Memory movement:
    x and y live in global memory (DRAM on a GPU, ordinary host memory under
    the interpreter). Each program instance loads a BLOCK_SIZE-element tile of
    each input into registers/SRAM, adds them there, and stores one tile of
    output back to global memory. Nothing is staged across program instances,
    so this kernel reads 2N elements and writes N.

Run with TRITON_INTERPRET=1 to execute on the CPU. Without a GPU present and
without that variable set, Triton has no backend to compile for and will fail.
"""

import os
import sys

INTERPRET = os.environ.get("TRITON_INTERPRET") == "1"

import torch
import triton
import triton.language as tl


@triton.jit
def add_kernel(x_ptr, y_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    # The last tile is partial whenever n_elements is not a multiple of
    # BLOCK_SIZE; the mask keeps those lanes from touching out-of-bounds memory.
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    tl.store(out_ptr + offsets, x + y, mask=mask)


def add(x: torch.Tensor, y: torch.Tensor, block_size: int = 128) -> torch.Tensor:
    out = torch.empty_like(x)
    n_elements = out.numel()
    grid = (triton.cdiv(n_elements, block_size),)
    add_kernel[grid](x, y, out, n_elements, BLOCK_SIZE=block_size)
    return out


def main() -> int:
    print(f"torch      {torch.__version__}")
    print(f"triton     {triton.__version__}")
    print(f"interpreter{'enabled' if INTERPRET else ' DISABLED':>10}")

    if not INTERPRET:
        print("\nTRITON_INTERPRET is not set to 1. On a CPU-only machine the "
              "kernel cannot be compiled. Re-run with TRITON_INTERPRET=1.")
        return 2

    torch.manual_seed(0)
    # Deliberately not a multiple of BLOCK_SIZE, so the masking path is exercised.
    n = 1000
    x = torch.randn(n)
    y = torch.randn(n)

    got = add(x, y)
    want = x + y
    max_abs_err = (got - want).abs().max().item()

    print(f"\nshape        ({n},)")
    print(f"max abs err  {max_abs_err:.3e}")

    if not torch.equal(got, want):
        print("\nFAIL: kernel output does not match torch exactly")
        return 1

    print("\nPASS: trivial kernel executed on CPU and matched torch bitwise")
    return 0


if __name__ == "__main__":
    sys.exit(main())
