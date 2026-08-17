"""Naive PyTorch implementations used as ground truth.

These are written for obviousness, not speed. Every one of them is a direct
transcription of the mathematical definition, so a reader can check them by
eye. The kernels are validated against these, never the other way round.

Precision policy
----------------
Every reference computes in fp32 internally and casts the result back to the
input dtype at the end. The Triton kernels do exactly the same thing (load,
`.to(tl.float32)`, compute, implicit cast on store), so the comparison
isolates kernel logic rather than measuring a difference in accumulation
precision.

Two distinct references exist for the composed pipeline, and the difference
between them is the whole reason the fused kernel is worth studying:

- `unfused_pipeline` rounds the intermediate to the input dtype, because in
  the two-kernel sequence that intermediate really is written to memory as a
  tensor of that dtype.
- `fused_pipeline` keeps the intermediate in fp32, because in the fused kernel
  it never leaves the on-chip tile.

In fp32 these coincide. In bf16 they do not, and the fused version is the
more accurate of the two.
"""

from __future__ import annotations

import torch


def rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """y = x / sqrt(mean(x^2) + eps) * weight, over the last dimension."""
    x32 = x.to(torch.float32)
    mean_square = x32.pow(2).mean(dim=-1, keepdim=True)
    y = x32 * torch.rsqrt(mean_square + eps) * weight.to(torch.float32)
    return y.to(x.dtype)


def softmax(x: torch.Tensor) -> torch.Tensor:
    """Numerically stabilised softmax over the last dimension."""
    x32 = x.to(torch.float32)
    shifted = x32 - x32.max(dim=-1, keepdim=True).values
    exponentiated = torch.exp(shifted)
    y = exponentiated / exponentiated.sum(dim=-1, keepdim=True)
    return y.to(x.dtype)


def unfused_pipeline(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """What the two-kernel sequence computes, intermediate rounding included."""
    intermediate = rmsnorm(x, weight, eps)  # materialised in x.dtype
    return softmax(intermediate)


def fused_pipeline(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """What the fused kernel computes: one fp32 chain, rounded once at the end."""
    x32 = x.to(torch.float32)
    mean_square = x32.pow(2).mean(dim=-1, keepdim=True)
    h = x32 * torch.rsqrt(mean_square + eps) * weight.to(torch.float32)
    shifted = h - h.max(dim=-1, keepdim=True).values
    exponentiated = torch.exp(shifted)
    y = exponentiated / exponentiated.sum(dim=-1, keepdim=True)
    return y.to(x.dtype)


def ground_truth(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """The pipeline in float64, returned in float64.

    Used to rank two low-precision implementations against each other. Both
    the fused kernel and the unfused sequence are approximations; comparing
    them only to one another says which is *different*, not which is *right*.
    This says which is right.

    The inputs are still the quantised x and weight, so this measures error
    introduced by the computation, not error already present in the inputs.
    """
    x64 = x.to(torch.float64)
    mean_square = x64.pow(2).mean(dim=-1, keepdim=True)
    h = x64 * torch.rsqrt(mean_square + eps) * weight.to(torch.float64)
    shifted = h - h.max(dim=-1, keepdim=True).values
    exponentiated = torch.exp(shifted)
    return exponentiated / exponentiated.sum(dim=-1, keepdim=True)
